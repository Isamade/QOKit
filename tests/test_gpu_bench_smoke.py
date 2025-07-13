import pytest, subprocess, sys, pathlib, importlib.util
import numba.cuda

@pytest.mark.skipif(not numba.cuda.is_available(), reason="GPU not available")
def test_gpu_bench_smoke(tmp_path):
    out = tmp_path / "bench.csv"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--Ns", "16", "--ps", "1", "--iters", "10",
         "--outfile", str(out)],
        capture_output=True, timeout=120
    )
    # Optional debug prints
    # print(proc.stdout.decode(), proc.stderr.decode())
    assert proc.returncode == 0, "benchmark script failed"
    assert out.exists() and out.stat().st_size > 0
