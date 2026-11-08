"""Standard library integration checks, also included in CTest."""

import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

binary, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=True)
prefix = struct.pack("<4sHHIQfff", b"SENS", 1, 24, 0, 0, 25.0, 0.1, 2.0)
payload = prefix + struct.pack("<I", zlib.crc32(prefix))
source, log, stats = root / "input.bin", root / "output.csv", root / "stats.json"
source.write_bytes(payload)
for input_arg, output_arg in ((str(source), str(log)), ("-", "-")):
    result = subprocess.run([str(binary), "--input", input_arg, "--output", output_arg,
                             "--stats", str(stats)], input=payload, capture_output=True, check=True)
    assert json.loads(stats.read_text())["accepted"] == 1
    if output_arg == "-":
        assert b"temperature" in result.stdout
for input_arg, output_arg, stats_arg in (
    (str(source), str(root / "missing/out.csv"), str(stats)),
    (str(source), str(log), str(root / "missing/stats.json")),
    (str(source), "/dev/full", str(stats)),
    (str(source), str(log), "/dev/full"),
    (str(root), str(log), str(stats)),
):
    result = subprocess.run([str(binary), "--input", input_arg, "--output", output_arg,
                             "--stats", stats_arg], capture_output=True)
    assert result.returncode != 0
