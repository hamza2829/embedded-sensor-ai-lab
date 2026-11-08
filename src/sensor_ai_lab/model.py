"""Small CPU autoencoder, static int8 ONNX export, and measured comparisons."""

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from scipy.stats import rankdata
from torch import nn

from sensor_ai_lab.analysis import FEATURE_NAMES, features
from sensor_ai_lab.sensors import Fault, simulate


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 4), nn.ReLU(),
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 12),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def examples(start_seed: int, runs: int, kind: str = "normal") -> tuple[np.ndarray, list[dict]]:
    if runs < 1:
        raise ValueError("runs must be positive")
    rows, identities = [], []
    for seed in range(start_seed, start_seed + runs):
        rng = np.random.default_rng(seed + 999999)
        faults = ()
        if kind == "stuck":
            faults = (Fault("stuck", 0, 256),)
        elif kind == "current_step":
            faults = (Fault("current_step", 0, 256, float(rng.uniform(0.12, 0.6))),)
        elif kind == "vibration_shift":
            faults = (Fault("vibration_shift", 0, 256, float(rng.uniform(35, 85))),)
        elif kind == "spike":
            magnitude = float(rng.uniform(0.15, 2.0))
            faults = (Fault("spike", 32, 1, magnitude), Fault("spike", 160, 1, magnitude))
        elif kind != "normal":
            raise ValueError(f"unknown model evaluation class: {kind}")
        extracted = features(simulate(seed, samples=256, faults=faults))
        rows.extend(extracted)
        identities.extend(dict(seed=seed, window=window, kind=kind, label=int(kind != "normal"))
                          for window in range(len(extracted)))
    return np.asarray(rows, dtype=np.float32), identities


def train_autoencoder(training: np.ndarray, validation: np.ndarray, epochs: int = 60) -> tuple:
    if epochs < 1 or len(training) < 2 or training.shape[1] != 12:
        raise ValueError("training needs positive epochs and at least two 12-feature rows")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    mean = training.mean(0)
    scale = training.std(0).clip(min=0.01)
    x = torch.from_numpy(((training - mean) / scale).astype(np.float32))
    val = torch.from_numpy(((validation - mean) / scale).astype(np.float32))
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = float(((model(x) - x) ** 2).mean().detach())
    generator = torch.Generator().manual_seed(42)
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        for indices in torch.randperm(len(x), generator=generator).split(64):
            loss = ((model(x[indices]) - x[indices]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            train_loss = float(((model(x) - x) ** 2).mean())
            val_loss = float(((model(val) - val) ** 2).mean())
        history.append(dict(epoch=epoch + 1, train_loss=train_loss, validation_loss=val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        final = float(((model(x) - x) ** 2).mean())
    return model, mean, scale, dict(initial_loss=initial, final_loss=final, history=history)


class Calibration(CalibrationDataReader):
    def __init__(self, inputs: np.ndarray):
        self.batches = iter({"features": batch} for batch in np.array_split(inputs, min(32, len(inputs))))

    def get_next(self):
        return next(self.batches, None)


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def scores(runtime: ort.InferenceSession, inputs: np.ndarray) -> np.ndarray:
    reconstruction = runtime.run(None, {"features": inputs})[0]
    return np.mean((reconstruction - inputs) ** 2, axis=1)


def metrics(labels: np.ndarray, values: np.ndarray, threshold: float) -> dict:
    predicted = values > threshold
    positive = labels == 1
    tp, fp = int(np.sum(predicted & positive)), int(np.sum(predicted & ~positive))
    fn, tn = int(np.sum(~predicted & positive)), int(np.sum(~predicted & ~positive))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    auc = None
    if n_positive and n_negative:
        auc = float((rankdata(values)[positive].sum() - n_positive * (n_positive + 1) / 2)
                    / (n_positive * n_negative))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=2 * precision * recall / max(precision + recall, 1e-12),
                accuracy=(tp + tn) / len(labels), roc_auc=auc)


def latency_pair(runtimes: dict, inputs: np.ndarray, repeats: int = 500) -> dict:
    timings = {name: [] for name in runtimes}
    for runtime in runtimes.values():
        for _ in range(50):
            runtime.run(None, {"features": inputs[:1]})
    for round_index in range(7):
        names = list(runtimes) if round_index % 2 == 0 else list(reversed(runtimes))
        for name in names:
            runtime = runtimes[name]
            for index in range(repeats):
                row = inputs[index % len(inputs):index % len(inputs) + 1]
                start = time.perf_counter_ns()
                runtime.run(None, {"features": row})
                timings[name].append((time.perf_counter_ns() - start) / 1000)
    return {name: dict(median_us=float(np.median(values)), p95_us=float(np.quantile(values, 0.95)),
                       samples=len(values), batch_size=1, warmup=50)
            for name, values in timings.items()}


def run_model_lab(output: Path, train_runs: int = 256, validation_runs: int = 64,
                  test_runs: int = 128, epochs: int = 60, latency_repeats: int = 500) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if train_runs > 10000 or validation_runs > 10000 or test_runs > 1000:
        raise ValueError("run counts would overlap reserved seed ranges")
    training, _ = examples(0, train_runs)
    validation, _ = examples(10000, validation_runs)
    started = time.perf_counter()
    model, mean, scale, training_report = train_autoencoder(training, validation, epochs)
    training_report["seconds"] = time.perf_counter() - started
    normalized = ((training - mean) / scale).astype(np.float32)
    float_path, int8_path = output / "float.onnx", output / "int8.onnx"
    torch.onnx.export(model, torch.zeros(1, 12), str(float_path), input_names=["features"],
                      output_names=["reconstruction"], dynamic_axes={"features": {0: "batch"},
                      "reconstruction": {0: "batch"}}, opset_version=17, dynamo=False)
    preprocessed = output / "preprocessed.onnx"
    quant_pre_process(str(float_path), str(preprocessed))
    preprocessed.replace(float_path)
    quantize_static(str(float_path), str(int8_path), Calibration(normalized),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True)
    for path in (float_path, int8_path):
        onnx.checker.check_model(onnx.load(path))
    quantized_graph = onnx.load(int8_path)
    int8_tensors = sum(item.data_type == onnx.TensorProto.INT8
                      for item in quantized_graph.graph.initializer)
    if not int8_tensors:
        raise ValueError("export did not contain int8 tensors")
    runtimes = {"float": session(float_path), "int8": session(int8_path)}
    with torch.inference_mode():
        reference = model(torch.from_numpy(normalized[:16])).numpy()
    parity = float(np.max(np.abs(reference - runtimes["float"].run(
        None, {"features": normalized[:16]})[0])))
    if parity > 1e-5:
        raise ValueError(f"PyTorch to ONNX parity error: {parity}")
    val = ((validation - mean) / scale).astype(np.float32)
    threshold = float(np.quantile(scores(runtimes["float"], val), 0.99))
    test_parts, identities = [], []
    for index, kind in enumerate(("normal", "stuck", "current_step", "vibration_shift", "spike")):
        part, ids = examples(20000 + index * 1000, test_runs, kind)
        test_parts.append(part)
        identities.extend(ids)
    inputs = ((np.concatenate(test_parts) - mean) / scale).astype(np.float32)
    labels = np.asarray([row["label"] for row in identities])
    values = {name: scores(runtime, inputs) for name, runtime in runtimes.items()}
    measured_latency = latency_pair(runtimes, inputs, latency_repeats)
    comparisons = {}
    for name, path in (("float", float_path), ("int8", int8_path)):
        by_class = {}
        for kind in sorted({row["kind"] for row in identities}):
            mask = np.asarray([row["kind"] == kind for row in identities])
            by_class[kind] = dict(rows=int(mask.sum()),
                                 alarm_rate=float(np.mean(values[name][mask] > threshold)))
        comparisons[name] = dict(bytes=path.stat().st_size, latency=measured_latency[name],
                                 quality=metrics(labels, values[name], threshold), by_class=by_class)
    float_prediction, int8_prediction = values["float"] > threshold, values["int8"] > threshold
    losses = (float_prediction == labels) & (int8_prediction != labels)
    gains = (float_prediction != labels) & (int8_prediction == labels)
    outcomes = [dict(**row, float_score=float(values["float"][index]),
                     int8_score=float(values["int8"][index]),
                     float_prediction=int(float_prediction[index]),
                     int8_prediction=int(int8_prediction[index]))
                for index, row in enumerate(identities)]
    report = dict(
        training=training_report, parameter_count=sum(item.numel() for item in model.parameters()),
        seed_ranges=dict(training=[0, train_runs - 1], validation=[10000, 10000 + validation_runs - 1],
                         test_start_seeds=[20000, 21000, 22000, 23000, 24000], runs_per_class=test_runs),
        feature_names=FEATURE_NAMES, mean=mean.tolist(), scale=scale.tolist(), threshold=threshold,
        threshold_protocol="99th percentile of float scores on separate normal validation runs; shared",
        quantization="static QDQ; signed int8 weights and activations; float inputs, outputs, biases",
        int8_initializer_count=int8_tensors, onnx_parity_max_abs=parity, models=comparisons,
        int8_losses=int(losses.sum()), int8_gains=int(gains.sum()),
        decision_disagreements=int(np.sum(float_prediction != int8_prediction)),
        int8_loss_examples=[outcomes[index] for index in np.flatnonzero(losses)[:10]],
        outcomes=outcomes,
        timing_protocol="CPUExecutionProvider; one thread; 7 alternating rounds; batch one; model only",
    )
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), output / "autoencoder.pt")
    np.savez(output / "evaluation_inputs.npz", normalized_features=inputs, labels=labels)
    return report
