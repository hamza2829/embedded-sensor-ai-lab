import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from sensor_ai_lab.analysis import FEATURE_NAMES, analyze, clean_resample, detect_faults, features
from sensor_ai_lab.bench import acquire, run_bench
from sensor_ai_lab.sensors import Fault, encode_frame, encode_stream, simulate


@pytest.mark.parametrize("pipe", [True, False])
def test_python_to_c_binary_round_trip(daq, tmp_path, pipe):
    frame = simulate(22, samples=256)
    log, stats = acquire(daq, encode_stream(frame), tmp_path, pipe)
    measured = pd.read_csv(log)
    for column in ["temperature", "vibration", "current"]:
        np.testing.assert_allclose(measured[column], frame[column], rtol=1e-6, atol=1e-6)
    assert (measured.acquired_ns > 1_000_000_000_000_000_000).all()
    assert measured.source_us.tolist() == frame.source_us.tolist()
    assert json.loads(stats.read_text())["accepted"] == 256


def test_fault_bench_detects_each_fault_and_clean_control(daq, tmp_path):
    result = run_bench(daq, tmp_path)
    assert result["passed"]
    assert len(result["cases"]) == 5
    assert result["cases"]["dropped_frames"]["acquisition"]["sequence_gaps"] == 4
    assert result["cases"]["bad_crc"]["acquisition"]["bad_crc"] == 3
    assert not any(result["cases"]["clean"]["alerts"].values())
    for name in result["cases"]:
        assert (tmp_path / name / "analysis/signals.png").stat().st_size > 1000
        assert (tmp_path / name / "analysis/spectrum.png").stat().st_size > 1000
        assert "Simulated measurement" in (tmp_path / name / "analysis/measurement.md").read_text()


def test_c_rejects_bad_header_values_order_and_truncation(daq, tmp_path):
    valid = encode_frame(0, 0, (25, 0, 2))
    header = b"NOPE" + valid[4:]
    bad_value = encode_frame(1, 5000, (float("nan"), 0, 2))
    duplicate = encode_frame(0, 5000, (25, 0, 2))
    log, stats = acquire(daq, valid + header + bad_value + duplicate + b"abc", tmp_path)
    result = json.loads(stats.read_text())
    assert result["accepted"] == 1
    assert result["bad_header"] == result["bad_values"] == result["out_of_order"] == 1
    assert result["truncated_bytes"] == 3
    assert len(pd.read_csv(log)) == 1


def test_csv_analysis_artifacts(daq, tmp_path):
    log, stats = acquire(daq, encode_stream(simulate(12, samples=256)), tmp_path)
    result = analyze(log, stats, tmp_path / "analysis")
    assert result["feature_windows"] == 2
    assert abs(result["mean_features"]["vibration_peak_hz"] - 25) < 1
    assert len(pd.read_csv(tmp_path / "analysis/features.csv").columns) == len(FEATURE_NAMES)


def test_resampling_cleaning_and_interpolation():
    raw = simulate(3, samples=256)
    raw.loc[10, "temperature"] = np.inf
    raw.loc[20, "current"] = 1000
    raw = raw.drop([30, 31])
    cleaned, quality = clean_resample(raw)
    assert len(cleaned) == 256
    assert quality["invalid_values"] == 2
    assert quality["interpolated_rows"] == 4
    assert np.isfinite(cleaned.to_numpy()).all()
    slower, _ = clean_resample(simulate(0, samples=256), 100)
    assert len(slower) == 128


@pytest.mark.parametrize("case", ["schema", "time", "nan_time", "large_gap", "edge_nan", "rate"])
def test_cleaning_rejects_broken_logs(case):
    raw = simulate(0, samples=256)
    rate = 200
    if case == "schema":
        raw = raw.drop(columns="current")
    elif case == "time":
        raw.loc[5, "source_us"] = 0
    elif case == "nan_time":
        raw["source_us"] = raw.source_us.astype(float)
        raw.loc[5, "source_us"] = np.nan
    elif case == "large_gap":
        raw = raw.drop(range(10, 30))
    elif case == "edge_nan":
        raw.loc[0, "current"] = np.nan
    else:
        rate = 201
    with pytest.raises(ValueError):
        clean_resample(raw, rate)


def test_rms_fft_known_signal_and_feature_guards():
    t = np.arange(128) / 200
    raw = pd.DataFrame(dict(temperature=np.ones(128) * 25,
                            vibration=np.sin(2 * np.pi * 25 * t), current=np.ones(128) * 2))
    result = features(raw)[0]
    assert result.shape == (12,)
    assert result[6] == pytest.approx(1 / np.sqrt(2), rel=1e-5)
    assert result[7] == pytest.approx(25)
    assert result[3] == result[11] == 0
    with pytest.raises(ValueError, match="window"):
        features(raw, window=2)
    raw.loc[0, "current"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        features(raw)


def test_stuck_and_spike_rules():
    raw = simulate(0, faults=(Fault("stuck", 100, 40), Fault("spike", 200, 1)))
    alerts = detect_faults(raw, {})
    assert alerts["stuck_sensor"] and alerts["spikes"]


def test_subprocess_failure_is_visible(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        acquire("/usr/bin/false", b"", tmp_path)
