#!/usr/bin/env python
"""
Compare end-to-end GPU performance *before* vs *after* steps ⑤-⑧.

CSV columns:
  N,p,t_before,t_after,percent_gain
Run on a qBraid Large-GPU node:
  python scripts/benchmark_gpu_before_after.py --Ns 16 20 25 --ps 1 5 10
"""
from __future__ import annotations
import argparse, time, itertools, csv, pathlib, numpy as np
from qokit.portfolio_optimization import get_problem
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from scipy.optimize import minimize


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------
def baseline_time(po, p, iters=30):
    """
    GPU backend but *no* vectorised energies, *no* batch optimiser.
    """
    obj = get_qaoa_portfolio_objective(
        po, p=p, device="gpu", precomputed_energies=None   # ❶ old path
    )
    x0 = np.random.random(2 * p)

    def f(th):
        return obj(th)

    t0 = time.perf_counter()
    minimize(f, x0, method="L-BFGS-B",
             options={"maxiter": iters, "disp": False})
    return time.perf_counter() - t0


def enhanced_time(po, p, iters=30, B=8):
    """
    Vectorised energies + batch optimiser + async stream (new path).
    """
    obj = get_qaoa_portfolio_objective(
        po, p=p, device="gpu",
        precomputed_energies="vectorized"                  # ❷ new path
    )
    x0 = np.random.random(2 * p)
    t0 = time.perf_counter()
    obj.batch_opt(x0, B=B, maxiter=iters)
    return time.perf_counter() - t0


def one_case(N, p, iters):
    po = get_problem(N=N, K=int(0.3 * N), q=0.7, pre="rule")

    t_before = baseline_time(po, p, iters=iters)
    t_after  = enhanced_time(po, p, iters=iters)

    gain = 100 * (t_before - t_after) / t_before
    return dict(N=N, p=p,
                t_before=round(t_before, 3),
                t_after=round(t_after, 3),
                percent_gain=round(gain, 1))


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--Ns", nargs="+", type=int, default=[16, 20, 25])
    pa.add_argument("--ps", nargs="+", type=int, default=[1, 5, 10])
    pa.add_argument("--iters", type=int, default=30,
                    help="Optimizer iterations for each timing run")
    pa.add_argument("--outfile", default="results/gpu_before_after.csv")
    args = pa.parse_args()

    rows = [one_case(N, p, args.iters)
            for N, p in itertools.product(args.Ns, args.ps)]

    out = pathlib.Path(args.outfile)
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=rows[0].keys()).writeheader()
        csv.DictWriter(fh, fieldnames=rows[0].keys()).writerows(rows)

    print("\nBenchmark complete  ➜", out, "\n")
    for r in rows:
        print(f"N={r['N']:<2} p={r['p']:<2} "
              f"before={r['t_before']:.3f}s  after={r['t_after']:.3f}s  "
              f"gain={r['percent_gain']:.1f}%")
