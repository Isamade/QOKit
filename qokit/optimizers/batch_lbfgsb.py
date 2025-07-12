import numpy as np
from scipy.optimize import minimize
from qokit.utils import async_stream,pinned_array

def batch_minimize_lbfgsb(fun_vec, x0, *, batch_size=8, maxiter=200):
    """
    L-BFGS-B wrapper that evaluates `batch_size` parameter vectors per
    objective call.

    Parameters
    ----------
    fun_vec : callable(theta_matrix:(B,dim)) -> (B,) energies
        MUST accept a 2-D array and return vectorised energies.
    x0 : (dim,) initial guess  (will be broadcast to all batches)
    """
    dim   = x0.size
    theta = np.tile(x0, batch_size)   # (B*dim,)
    def f(flat):
        B = flat.size // dim
        with async_stream():  # NEW
            theta_mat = pinned_array(flat.reshape(B, dim))
        return float(fun_vec(theta_mat).mean())

    return minimize(f, theta, method="L-BFGS-B",
                    options={"maxiter": maxiter})
