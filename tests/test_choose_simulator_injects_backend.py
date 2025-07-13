from functools import partial
from qokit.fur import choose_simulator, choose_simulator_xyring
from qokit.sim_backend import get_backend

def test_backend_propagates_through_choose():
    bak = get_backend("cpu")            # any AerSimulator instance
    SimCls = choose_simulator("fast", backend=bak)
    assert isinstance(SimCls, partial)
    sim = SimCls(2, costs=[0,0,0,0], terms=None)
    assert sim.backend is bak           # attribute stored
