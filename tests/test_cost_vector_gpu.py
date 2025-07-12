import pytest
import importlib, numpy as np
from qokit.portfolio_optimization import brute_force_cost_vector, cost_vector_gpu
import numba.cuda
# ── Skip when either CuPy is missing *or* CUDA runtime is absent ──


@pytest.mark.skipif(not numba.cuda.is_available(), reason="GPU not available")
def test_gpu_matches_cpu():
    N = 8
    mu  = np.ones(N)
    cov = np.eye(N)
    q   = 0.7
    bit_mat = cp.asarray(np.arange(2**N)[:, None] >> np.arange(N) & 1)
    gpu = cost_vector_gpu(bit_mat, mu, cov, q).get()
    cpu = brute_force_cost_vector({"mu": mu, "cov": cov, "q": q, "N": N})
    np.testing.assert_allclose(gpu, cpu, rtol=1e-12, atol=1e-12)
