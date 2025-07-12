import pytest, importlib.util, numpy as np
from qokit.portfolio_optimization import get_problem
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from qokit.sim_backend import get_backend
import importlib.util, pytest
from qiskit_aer import AerSimulator

# Skip on CI nodes without GPU libs
gpu_available = "statevector_gpu" in AerSimulator().available_methods()
pytestmark = pytest.mark.skipif(not gpu_available,
                                reason="qiskit-aer-gpu not installed")





def test_gpu_backend_runs():
    backend_gpu = get_backend("gpu")          # statevector_gpu
    obj = get_qaoa_portfolio_objective(
        get_problem(N=4, K=1, q=0.7, pre="rule"),
        p=1,
        device="gpu",                         # high-level flag
        backend=backend_gpu,                  # low-level override
    )
    val = obj(np.array([0.1, 0.2]))
    assert isinstance(val, float)
    assert obj.backend.configuration().method == "statevector_gpu"
