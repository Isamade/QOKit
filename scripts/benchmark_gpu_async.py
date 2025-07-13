#!/usr/bin/env python
"""
Step ⑧: compare batch optimiser WITH vs WITHOUT async stream.
"""

import csv, pathlib, itertools, time, numpy as np
from qokit.portfolio_optimization import get_problem
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective

def run_case(N,p,async_on):
    po = get_problem(N=N, K=int(0.3*N), q=0.7, pre="rule")
    obj = get_qaoa_portfolio_objective(po, p=p,
                                       precomputed_energies="vectorized",
                                       device="gpu")
    x0  = np.random.random(2*p)
    if not async_on:
        # monkey-patch to disable async for baseline
        from qokit.utils import cuda_stream as cs
        cs.async_stream = lambda: (yield)
    t=time.perf_counter(); _=obj.batch_opt(x0,B=8,maxiter=50); return time.perf_counter()-t

def grid():
    rows=[]
    for N,p in itertools.product([20,25],[5,10]):
        t0 = run_case(N,p,async_on=False)
        t1 = run_case(N,p,async_on=True)
        rows.append(dict(N=N,p=p,t_no_async=round(t0,3),
                         t_async=round(t1,3),
                         gain=round(100*(t0-t1)/t0,1)))
    return rows

if __name__=="__main__":
    rows=grid()
    out=pathlib.Path("results");out.mkdir(exist_ok=True)
    fn=out/"gpu_async_gain.csv"
    with fn.open("w",newline="") as fh:
        import csv; csv.DictWriter(fh,fieldnames=rows[0].keys()).writerows(rows)
    print("CSV →",fn)
