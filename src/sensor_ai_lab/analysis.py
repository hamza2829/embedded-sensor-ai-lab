"""Data cleaning, fault rules, resampling, spectral features, and run reports."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from sensor_ai_lab.sensors import CHANNELS, SAMPLE_RATE

FEATURE_NAMES = [f"{channel}_{feature}" for channel in CHANNELS
                 for feature in ("mean", "std", "rms", "peak_hz")]


def clean_resample(raw: pd.DataFrame, sample_rate: int = SAMPLE_RATE) -> tuple[pd.DataFrame, dict]:
    if sample_rate < 1 or sample_rate > SAMPLE_RATE:
        raise ValueError("resampling rate must lie in [1, 200] Hz")
    if not {"source_us", *CHANNELS}.issubset(raw.columns) or len(raw) < 2:
        raise ValueError("log requires source_us and three sensor channels, with at least two rows")
    times = pd.to_numeric(raw["source_us"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("source timestamps must be finite, nonnegative, and strictly increasing")
    values = raw[CHANNELS].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    limits = {"temperature": (-50, 150), "vibration": (-100, 100), "current": (-50, 100)}
    for channel, (low, high) in limits.items():
        values.loc[~values[channel].between(low, high), channel] = np.nan
    invalid = int(values.isna().sum().sum())
    values.index = pd.to_timedelta(times, unit="us")
    resampled = values.resample(pd.to_timedelta(1 / sample_rate, unit="s")).mean()
    missing = int(resampled.isna().any(axis=1).sum())
    cleaned = resampled.interpolate(method="linear", limit=8, limit_area="inside")
    if cleaned.isna().any().any():
        raise ValueError("unfillable gaps: interpolation is limited to eight interior samples")
    return cleaned, dict(raw_rows=len(raw), resampled_rows=len(cleaned),
                         invalid_values=invalid, interpolated_rows=missing, sample_rate=sample_rate)


def detect_faults(raw: pd.DataFrame, stats: dict) -> dict:
    alerts = dict(dropped_frames=stats.get("sequence_gaps", 0) > 0,
                  bad_crc=stats.get("bad_crc", 0) > 0, stuck_sensor=False, spikes=False)
    for channel in CHANNELS:
        series = pd.to_numeric(raw[channel], errors="coerce")
        equal = series.diff().abs().fillna(1).lt(1e-10)
        groups = (~equal).cumsum()
        if int(equal.groupby(groups).sum().max()) >= 31:
            alerts["stuck_sensor"] = True
        center = series.rolling(31, center=True, min_periods=8).median()
        residual = (series - center).abs()
        mad = residual.rolling(31, center=True, min_periods=8).median()
        floor = {"temperature": 1.0, "vibration": 2.0, "current": 1.0}[channel]
        if (residual > np.maximum(8 * 1.4826 * mad, floor)).any():
            alerts["spikes"] = True
    return alerts


def features(frame: pd.DataFrame, window: int = 128, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if window < 8 or len(frame) < window or sample_rate < 1:
        raise ValueError("features require a window of at least eight and enough samples")
    values = frame[CHANNELS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("features require finite cleaned values")
    results = []
    for start in range(0, len(values) - window + 1, window):
        chunk = values[start:start + window]
        row = []
        for column in chunk.T:
            frequencies, spectrum = signal.periodogram(column, fs=sample_rate, detrend="linear")
            peak = (float(frequencies[1 + np.argmax(spectrum[1:])])
                    if np.std(column) > 1e-10 else 0.0)
            row.extend([np.mean(column), np.std(column), np.sqrt(np.mean(column ** 2)), peak])
        results.append(row)
    return np.asarray(results, dtype=np.float32)


def analyze(log: Path, stats_path: Path, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(log)
    stats = json.loads(Path(stats_path).read_text())
    alerts = detect_faults(raw, stats)
    cleaned, quality = clean_resample(raw)
    extracted = features(cleaned)
    pd.DataFrame(extracted, columns=FEATURE_NAMES).to_csv(output / "features.csv", index=False)
    cleaned.to_csv(output / "cleaned.csv", index_label="source_time")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for axis, channel in zip(axes, CHANNELS, strict=True):
        axis.plot(raw["source_us"] / 1e6, raw[channel], linewidth=0.8)
        axis.set_ylabel(channel)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Simulated sensor time (s)")
    fig.tight_layout()
    fig.savefig(output / "signals.png", dpi=130)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 3))
    frequency, power = signal.periodogram(cleaned["vibration"], fs=SAMPLE_RATE)
    axis.semilogy(frequency, np.maximum(power, 1e-12))
    axis.set(xlabel="Frequency (Hz)", ylabel="Vibration PSD", title="Cleaned vibration spectrum")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "spectrum.png", dpi=130)
    plt.close(fig)
    report = dict(acquisition=stats, cleaning=quality, alerts=alerts,
                  feature_windows=len(extracted), mean_features=dict(zip(
                      FEATURE_NAMES, extracted.mean(axis=0).astype(float).tolist(), strict=True)))
    (output / "measurement.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Simulated measurement report", "", "No hardware or real lab measurements.", "",
             f"Accepted frames: {stats['accepted']}. Feature windows: {len(extracted)}.", "",
             f"Cleaning: `{quality}`", "", f"Fault alerts: `{alerts}`", "",
             "| Feature | Mean over windows |", "| :--- | ---: |"]
    lines += [f"| {key} | {value:.6g} |" for key, value in report["mean_features"].items()]
    lines += ["", "![Sensor signals](signals.png)", "", "![Vibration spectrum](spectrum.png)"]
    (output / "measurement.md").write_text("\n".join(lines) + "\n")
    return report
