import struct
import zlib

import numpy as np
import pandas as pd
import pytest

from sensor_ai_lab.sensors import FRAME, FRAME_SIZE, Fault, encode_frame, encode_stream, simulate


def test_seed_determinism_noise_and_drift():
    pd.testing.assert_frame_equal(simulate(2), simulate(2))
    assert not simulate(2).equals(simulate(3))
    frame = simulate(10)
    assert frame.temperature.std() > 0.02
    assert frame.current.std() > 0.02
    assert frame.vibration.std() > 0.2
    assert np.diff(frame.source_us).tolist() == [5000] * (len(frame) - 1)


@pytest.mark.parametrize("kind", ["stuck", "spike", "drift", "current_step", "vibration_shift"])
def test_injected_signal_faults(kind):
    normal = simulate(5)
    changed = simulate(5, faults=(Fault(kind, 128, 64, 12),))
    assert not normal.equals(changed)
    pd.testing.assert_frame_equal(normal.iloc[:128], changed.iloc[:128])
    pd.testing.assert_frame_equal(normal.iloc[192:], changed.iloc[192:])


def test_wire_format_and_crc():
    payload = encode_frame(0x01020304, 0x1020304050607080, (25.5, -0.25, 2.0))
    assert len(payload) == FRAME_SIZE == 36
    assert payload[:8] == b"SENS\x01\x00\x18\x00"
    assert FRAME.unpack(payload[:32]) == (b"SENS", 1, 24, 0x01020304,
                                         0x1020304050607080, 25.5, -0.25, 2.0)
    assert struct.unpack("<I", payload[32:])[0] == zlib.crc32(payload[:32])


def test_transport_fault_encoding():
    frame = simulate(0, samples=10)
    normal = encode_stream(frame)
    assert len(encode_stream(frame, (Fault("drop", 2, 3),))) == len(normal) - 108
    damaged = encode_stream(frame, (Fault("bad_crc", 2, 1),))
    assert normal[:72] == damaged[:72]
    assert normal[108:] == damaged[108:]
    assert normal[72:108] != damaged[72:108]


@pytest.mark.parametrize("kwargs", [dict(seed=-1), dict(seed=0, samples=1),
                                    dict(seed=0, faults=(Fault("unknown"),)),
                                    dict(seed=0, faults=(Fault("spike", -1),)),
                                    dict(seed=0, faults=(Fault("spike", 2000, 200),))])
def test_invalid_simulation(kwargs):
    with pytest.raises(ValueError):
        simulate(**kwargs)
