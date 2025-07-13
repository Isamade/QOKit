import numpy as np
from qokit.simulator.fast_cpu import FastCPUSimulator

def test_fast_cpu_accepts_backend_none():
    sims = FastCPUSimulator(
        N= 4,
        costs=np.zeros(16),
        terms=None,
        backend=None,                 # <- new kwarg, should not break
    )
    sv = sims.simulate_qaoa([0.1], [0.2], None)
    assert np.isclose(np.linalg.norm(sv), 1.0)      # valid state-vector
