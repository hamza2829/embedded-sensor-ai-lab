import os
from pathlib import Path

import pytest

from sensor_ai_lab.model import run_model_lab


@pytest.fixture(scope="session")
def daq():
    path = Path(os.environ.get("DAQ_BINARY", "build/src/daq/daq")).resolve()
    if not path.is_file():
        pytest.fail(f"C acquisition executable is required: {path}; build it with CMake first")
    return path


@pytest.fixture(scope="session")
def exported(tmp_path_factory, daq):
    output = tmp_path_factory.mktemp("model")
    report = run_model_lab(output, train_runs=16, validation_runs=4, test_runs=3,
                           epochs=18, latency_repeats=4, daq=daq)
    return output, report
