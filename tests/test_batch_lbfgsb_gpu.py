import pytest, importlib, numpy as np
from qokit.portfolio_optimization import get_problem
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
import numba.cuda



@pytest.mark.skipif(not numba.cuda.is_available(), reason="GPU not available")
def test_batch_opt_reduces_energy():
    po  = get_problem(N=10, K=3, q=0.7, pre="rule")
    obj = get_qaoa_portfolio_objective(po, p=2,
                                       precomputed_energies="vectorized",
                                       device="gpu")
    x0  = np.random.random(4)
    e0  = obj(x0)
    res = obj.batch_opt(x0, B=8, maxiter=20)
    assert res.fun <= e0
