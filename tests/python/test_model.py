import json

import numpy as np
import onnx
import pytest
import torch

from sensor_ai_lab.model import (
    Autoencoder,
    Calibration,
    examples,
    metrics,
    run_model_lab,
    train_autoencoder,
)


def test_autoencoder_training_reduces_loss():
    training, _ = examples(0, 16)
    validation, _ = examples(10000, 4)
    model, mean, scale, report = train_autoencoder(training, validation, epochs=25)
    assert report["final_loss"] < report["initial_loss"] * 0.7
    assert model(torch.zeros(2, 12)).shape == (2, 12)
    assert mean.shape == scale.shape == (12,)
    assert (scale >= 0.01).all()


def test_export_quantization_quality_and_actual_size(exported):
    output, report = exported
    assert report["onnx_parity_max_abs"] < 1e-5
    assert report["int8_initializer_count"] > 0
    assert len(report["outcomes"]) == 30
    assert report["decision_disagreements"] == report["int8_losses"] + report["int8_gains"]
    assert json.loads((output / "comparison.json").read_text()) == report
    assert (output / "evaluation_inputs.npz").exists()
    for name in ["float", "int8"]:
        path = output / f"{name}.onnx"
        onnx.checker.check_model(onnx.load(path))
        item = report["models"][name]
        assert item["bytes"] == path.stat().st_size
        assert item["latency"]["median_us"] > 0
        assert 0 <= item["quality"]["f1"] <= 1
        assert 0 <= item["quality"]["roc_auc"] <= 1
        assert sum(item["quality"][key] for key in ["tp", "fp", "fn", "tn"]) == 30
    assert report["seed_ranges"]["training"][1] < report["seed_ranges"]["validation"][0]


def test_metrics_known_confusion_and_auc():
    result = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.6, 0.9]), 0.5)
    assert result["tp"] == 2 and result["fp"] == 1 and result["fn"] == 0 and result["tn"] == 1
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["f1"] == pytest.approx(0.8)
    assert metrics(np.zeros(3), np.zeros(3), 1)["roc_auc"] is None


def test_calibration_finishes_and_guards(tmp_path):
    reader = Calibration(np.zeros((2, 12), dtype=np.float32))
    assert reader.get_next()["features"].shape == (1, 12)
    assert reader.get_next() is not None
    assert reader.get_next() is None
    with pytest.raises(ValueError, match="positive"):
        examples(0, 0)
    with pytest.raises(ValueError, match="unknown"):
        examples(0, 1, "bad")
    with pytest.raises(ValueError, match="positive"):
        train_autoencoder(np.zeros((2, 12)), np.zeros((2, 12)), epochs=0)
    with pytest.raises(ValueError, match="overlap"):
        run_model_lab(tmp_path, train_runs=10001)
    assert Autoencoder()(torch.zeros(1, 12)).shape == (1, 12)
