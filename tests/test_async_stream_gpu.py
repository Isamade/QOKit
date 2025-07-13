import pytest
import importlib.util
import numpy as np
from qokit.utils import async_stream, pinned_array

gpu = importlib.util.find_spec("cupy") is not None
pytestmark = pytest.mark.skipif(not gpu, reason="CuPy not installed")

def test_async_does_not_crash():
    arr = np.arange(8, dtype=np.float64)
    with async_stream():
        pa = pinned_array(arr)
        assert pa.shape == arr.shape
