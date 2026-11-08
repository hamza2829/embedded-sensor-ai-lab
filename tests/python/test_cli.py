import pytest

from sensor_ai_lab.bench import acquire
from sensor_ai_lab.cli import main


def test_cli_simulate_and_analyze(daq, tmp_path):
    frames = tmp_path / "frames.bin"
    assert main(["simulate", "--output", str(frames), "--samples", "256"]) == 0
    log, stats = acquire(daq, frames.read_bytes(), tmp_path / "daq")
    assert main(["analyze", str(log), "--stats", str(stats),
                 "--output", str(tmp_path / "analysis")]) == 0


def test_cli_bench_and_model_dispatch(monkeypatch, tmp_path, exported):
    monkeypatch.setattr("sensor_ai_lab.cli.run_bench", lambda *args: {"passed": True})
    monkeypatch.setattr("sensor_ai_lab.cli.run_model_lab", lambda *args, **kwargs: exported[1])
    assert main(["bench", "--daq", "/unused", "--output", str(tmp_path)]) == 0
    assert main(["train-export", "--daq", "/unused", "--output", str(tmp_path),
                 "--epochs", "2"]) == 0


def test_cli_error_message(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["simulate", "--seed", "-1", "--output", str(tmp_path / "bad")])
    assert exc.value.code == 2
    assert "error: seed must be nonnegative" in capsys.readouterr().err
