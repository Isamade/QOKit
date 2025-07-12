# tests/test_gpu_bench_smoke.py
import pytest, subprocess, sys, pathlib
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# check Aer GPU backend
gpu_backend_ok = "statevector_gpu" in AerSimulator().available_methods()

# check CuPy runtime
try:
    import cupy as cp
    _ = cp.cuda.runtime.getDeviceCount()          # raises if DLL missing
    cupy_ok = True
except Exception:
    cupy_ok = False

pytestmark = pytest.mark.skipif(
    not (gpu_backend_ok and cupy_ok),
    reason="GPU backend or CuPy runtime not available",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "benchmark_gpu_before_after.py"

def test_gpu_bench_smoke(tmp_path):
    out = tmp_path / "bench.csv"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--Ns", "16", "--ps", "1", "--iters", "10",
         "--outfile", str(out)],
        capture_output=True, timeout=120
    )
    assert proc.returncode == 0, "benchmark script failed"
    assert out.exists() and out.stat().st_size > 0
