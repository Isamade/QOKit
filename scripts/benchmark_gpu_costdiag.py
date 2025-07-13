#!/usr/bin/env python
"""
Compare GPU run **without** CuPy diag (v0.4.0) vs **with** (HEAD).
Run on Large-GPU; outputs CSV + PNG.
"""
from __future__ import annotations
import argparse, csv, pathlib, itertools, time
import numpy as np
from qokit.portfolio_optimization import get_problem
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective

def timed(fn):
    t0=time.perf_counter(); fn(); return time.perf_counter()-t0

def one_case(N,p,diag):
    po = get_problem(N=N,K=int(0.3*N),q=0.7,pre="rule")
    kw = dict(p=p,device="gpu",
              precomputed_energies="vectorized" if diag else None)
    obj = get_qaoa_portfolio_objective(po, **kw)
    t = timed(lambda: obj(np.random.default_rng(0).random(2*p)))
    return t

def run_grid(Ns,ps):
    rows=[]
    for N,p in itertools.product(Ns,ps):
        t_before = one_case(N,p,diag=False)
        t_after  = one_case(N,p,diag=True)
        gain = 100*(t_before-t_after)/t_before
        rows.append(dict(N=N,p=p,t_before=round(t_before,3),
                         t_after=round(t_after,3),
                         percent_gain=round(gain,1)))
    return rows

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--Ns",nargs="+",type=int,default=[16,20,25])
    pa.add_argument("--ps",nargs="+",type=int,default=[1,5,10])
    args=pa.parse_args()
    rows=run_grid(args.Ns,args.ps)
    out=pathlib.Path("results");out.mkdir(exist_ok=True)
    csvf=out/"gpu_costdiag_gain.csv"
    with csvf.open("w",newline="") as fh:
        csv.DictWriter(fh,fieldnames=rows[0].keys()).writerows(rows)
    print("CSV →",csvf)
