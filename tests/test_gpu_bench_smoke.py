# tests/test_gpu_benchmark_smoke.py
import pytest, subprocess, sys, pathlib, importlib.util, os

gpu = importlib.util.find_spec("cupy") is not None
pytestmark = pytest.mark.skipif(not gpu, reason="GPU libs not installed")

def test_gpu_bench_smoke(tmp_path):
    out = tmp_path/"bench.csv"
    subprocess.run(
        [sys.executable, "scripts/benchmark_gpu_before_after.py",
         "--Ns", "16", "--ps", "1", "--iters", "10",
         "--outfile", str(out)],
        check=True, timeout=120
    )
    assert out.exists() and out.stat().st_size > 0
