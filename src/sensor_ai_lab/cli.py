"""CLI for sensor generation, analysis, the fault bench, and the model lab."""

import argparse
import json
from pathlib import Path

from sensor_ai_lab.analysis import analyze
from sensor_ai_lab.bench import run_bench
from sensor_ai_lab.model import run_model_lab
from sensor_ai_lab.sensors import encode_stream, simulate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulated embedded sensor AI measurement chain")
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("simulate")
    generation.add_argument("--seed", type=int, default=123)
    generation.add_argument("--samples", type=int, default=2048)
    generation.add_argument("--output", type=Path, required=True)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("log", type=Path)
    analysis.add_argument("--stats", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    bench = commands.add_parser("bench")
    bench.add_argument("--daq", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--seed", type=int, default=123)
    training = commands.add_parser("train-export")
    training.add_argument("--daq", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        if args.command == "simulate":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(encode_stream(simulate(args.seed, args.samples)))
            print(args.output)
        elif args.command == "analyze":
            print(json.dumps(analyze(args.log, args.stats, args.output), indent=2))
        elif args.command == "bench":
            print(json.dumps(run_bench(args.daq, args.output, args.seed), indent=2))
        else:
            result = run_model_lab(args.output, epochs=args.epochs, daq=args.daq)
            print(json.dumps({key: value for key, value in result.items()
                              if key not in {"outcomes", "training"}}, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
