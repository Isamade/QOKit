#!/usr/bin/env python
"""
Compare per-θ optimiser vs new batch L-BFGS-B on GPU.
"""
import csv, pathlib, time, itertools
import numpy as np
from qokit.portfolio_optimization import get_problem
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective

def one(N,p,B):
    po = get_problem(N=N, K=int(0.3*N), q=0.7, pre="rule")
    obj = get_qaoa_portfolio_objective(po, p=p,
                                       precomputed_energies="vectorized",
                                       device="gpu")
    x0  = np.random.random(2*p)
    t1  = time.perf_counter(); _ = obj.batch_opt(x0,B=B,maxiter=50); t1=time.perf_counter()-t1
    t0  = time.perf_counter(); _ = obj(x0);                      t0=time.perf_counter()-t0
    return dict(N=N,p=p,B=B,t_per_theta=round(t0,4),
                t_batch=round(t1,4),
                gain=round(100*(t0*B - t1)/(t0*B),1))

if __name__=="__main__":
    rows=[one(N,p,B=8) for N,p in itertools.product([20,25], [3,5,10])]
    out=pathlib.Path("results");out.mkdir(exist_ok=True)
    csvf=out/"gpu_batch_gain.csv"
    with csvf.open("w",newline="") as fh:
        csv.DictWriter(fh,fieldnames=rows[0].keys()).writerows(rows)
    print("CSV →",csvf)
