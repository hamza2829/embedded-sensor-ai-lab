"""Seeded sensor signals and an explicitly specified binary wire format."""

import struct
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

CHANNELS = ["temperature", "vibration", "current"]
SAMPLE_RATE = 200
FRAME = struct.Struct("<4sHHIQfff")
FRAME_SIZE = 36


@dataclass(frozen=True)
class Fault:
    kind: str
    start: int = 128
    length: int = 64
    magnitude: float = 8.0


def simulate(seed: int, samples: int = 2048, faults: tuple[Fault, ...] = ()) -> pd.DataFrame:
    if seed < 0 or samples < 2:
        raise ValueError("seed must be nonnegative and samples at least two")
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / SAMPLE_RATE
    phase = rng.uniform(0, 2 * np.pi)
    temperature = 25 + rng.uniform(-0.8, 0.8) + rng.uniform(-0.02, 0.04) * t
    temperature = temperature + rng.normal(0, 0.03, samples)
    vibration = 0.4 * np.sin(2 * np.pi * 25 * t + phase)
    vibration += 0.08 * np.sin(2 * np.pi * 62 * t) + rng.normal(0, 0.025, samples)
    current = 2 + rng.uniform(-0.12, 0.12) + 0.08 * np.sin(2 * np.pi * 3 * t + phase)
    current += rng.uniform(-0.002, 0.002) * t + rng.normal(0, 0.015, samples)
    frame = pd.DataFrame(dict(sequence=np.arange(samples, dtype=np.uint32),
                              source_us=np.arange(samples, dtype=np.uint64) * 5000,
                              temperature=temperature, vibration=vibration, current=current))
    for fault in faults:
        if fault.start < 0 or fault.length < 1 or fault.start + fault.length > samples:
            raise ValueError("fault range must lie within the simulated run")
        indices = frame.index[fault.start:fault.start + fault.length]
        if fault.kind == "stuck":
            frame.loc[indices, "temperature"] = frame.loc[fault.start, "temperature"]
        elif fault.kind == "spike":
            frame.loc[indices, "current"] += fault.magnitude
            frame.loc[indices, "vibration"] += fault.magnitude * 0.6
        elif fault.kind == "drift":
            frame.loc[indices, "temperature"] += np.linspace(0, fault.magnitude, len(indices))
        elif fault.kind == "current_step":
            frame.loc[indices, "current"] += fault.magnitude
        elif fault.kind == "vibration_shift":
            frame.loc[indices, "vibration"] = (
                0.4 * np.sin(2 * np.pi * fault.magnitude * t[indices] + phase)
                + rng.normal(0, 0.025, len(indices))
            )
        elif fault.kind not in {"drop", "bad_crc"}:
            raise ValueError(f"unknown fault: {fault.kind}")
    return frame


def encode_frame(sequence: int, source_us: int, values: tuple[float, float, float]) -> bytes:
    prefix = FRAME.pack(b"SENS", 1, 24, sequence, source_us, *values)
    return prefix + struct.pack("<I", zlib.crc32(prefix))


def encode_stream(frame: pd.DataFrame, faults: tuple[Fault, ...] = ()) -> bytes:
    dropped, corrupted = set(), set()
    for fault in faults:
        if fault.kind in {"drop", "bad_crc"}:
            selected = set(range(fault.start, fault.start + fault.length))
            (dropped if fault.kind == "drop" else corrupted).update(selected)
    chunks = []
    for row in frame.itertuples(index=False):
        if row.sequence in dropped:
            continue
        payload = encode_frame(row.sequence, row.source_us,
                               (row.temperature, row.vibration, row.current))
        if row.sequence in corrupted:
            payload = payload[:32] + bytes([payload[32] ^ 1]) + payload[33:]
        chunks.append(payload)
    return b"".join(chunks)
