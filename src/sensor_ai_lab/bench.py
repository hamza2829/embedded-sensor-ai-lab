"""Run the real C acquisition executable against seeded simulated frames."""

import json
import subprocess
from pathlib import Path

import pandas as pd

from sensor_ai_lab.analysis import analyze, detect_faults
from sensor_ai_lab.sensors import Fault, encode_stream, simulate


def acquire(daq: Path, payload: bytes, output: Path, pipe: bool = True) -> tuple[Path, Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log, stats = output / "samples.csv", output / "daq_stats.json"
    source = "-"
    if not pipe:
        binary = output / "frames.bin"
        binary.write_bytes(payload)
        source = str(binary)
    command = [str(Path(daq).resolve()), "--input", source, "--output", str(log), "--stats", str(stats)]
    subprocess.run(command, input=payload if pipe else None, check=True, capture_output=True)
    return log, stats


def run_bench(daq: Path, output: Path, seed: int = 123) -> dict:
    cases = {
        "clean": (), "dropped_frames": (Fault("drop", 200, 4),),
        "bad_crc": (Fault("bad_crc", 300, 3),),
        "stuck_sensor": (Fault("stuck", 400, 96),),
        "spikes": (Fault("spike", 600, 1),),
    }
    results = {}
    for name, faults in cases.items():
        frame = simulate(seed, faults=faults)
        log, stats = acquire(daq, encode_stream(frame, faults), Path(output) / name, pipe=name != "clean")
        alerts = detect_faults(pd.read_csv(log), json.loads(stats.read_text()))
        passed = not any(alerts.values()) if name == "clean" else alerts[name]
        results[name] = dict(passed=bool(passed), alerts=alerts,
                             acquisition=json.loads(stats.read_text()))
        analyze(log, stats, Path(output) / name / "analysis")
    report = dict(seed=seed, cases=results, passed=all(item["passed"] for item in results.values()))
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "bench.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError(f"fault bench failed: {results}")
    return report
