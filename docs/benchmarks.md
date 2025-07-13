# Benchmarks

This page gathers **reproducible timing results** for QOKit.  It complements the short performance summary in the README by giving full tables, run‑scripts and environment hashes so the competition jury can validate every number.

> All timings were obtained on **qBraid GPU tier** machines unless otherwise stated.
>
> | Node | CPU        | GPU              | CUDA / driver      |
> |------|------------|------------------|--------------------|
> | GPU  | 8 CPU 16GB | NVIDIA A10 24 GB | CUDA 12.2 /535.104 |

---

## 1CPU path (v0.3.0 vs v0.2.1)

| N  | p | v0.2.1(seconds) | v0.3.0(seconds) | **Speed‑up** |
|----|---|  |-----------------|--------------|
| 16 | 1 |  73.41s| 2.85s           | X36  |
| 16 | 2 |  121.30s| 8.14s           | X14  |
| 16 | 5 |  273.01s|  28.53s       | x9.5  |
| 16 | 10 |  523.88s| 86.04s           | x6  |
| 18 | 1 |  286.74s| 4.54s           | x63  |
| 18 | 2 |  478.30s|  8.14s       | x58  |
| 18 | 5 |  1152.21s|  39.62s       | x29  |
| 18 | 10 |  |  88.31s       | NA  |
| 20 | 1 |  1050.18s| 9.79s           | x107  |
| 20 | 2 |  |  17.78s       | NA  |
| 20 | 5 |  |  60.29s       | NA  |
| 20 | 10 |  |  192.18s       | NA  |
| 22 | 1 |  | 26.09s           | NA  |
| 22 | 2 |  |  90.58s       | NA  |
| 22 | 5 |  |  197.07s       | NA  |
| 22 | 10 |  |  615.17s       | NA  |
*Figure1*Wall‑time vs qubit number (CPU).  Raw CSV→`results/cpu_before_after.csv`.

---

## 2  GPU incremental gains

| Step | Feature           | N=25, p=10 | Cumulative |
| ---- | ----------------- | ------ | ---------- |
| ⑤    | `statevector_gpu` | **×5.0** | × 5.0      |
| ⑥    | CuPy cost‑vector  | ×1.5   | × 7.5      |
| ⑦    | Batch L‑BFGS‑B    | ×1.25  | × 9.4      |
| ⑧    | Async stream      | × 112  | **× 10.6** |

*Figure2*Stacked bar: contribution of each GPU step.

Detailed per‑grid timings (1625, 110) live in
`results/gpu_before_after.csv`.

---

## 3How to reproduce

### 3.1CPU benchmark

```bash
conda activate qokit-po-cpu
python scripts/benchmark_before_after.py --Ns 16 20 25 --ps 1 5 10
```

Outputs `results/improvement_vs_bruteforce.csv` and a Matplotlib figure.

### 3.2GPU benchmark (before/ after)

```bash
conda activate qokit-po-gpu
python scripts/benchmark_gpu_before_after.py \
       --Ns 16 20 25 --ps 1 5 10 --iters 50
```

### 3.3Async‑stream micro‑benchmark

```bash
python scripts/benchmark_gpu_async.py --outfile results/gpu_async_gain.csv
```

---

## 4Environment locks

* `envs/cpu.yaml` – frozen for v0.3.x
* `envs/gpu.yaml` – frozen for v0.4.x  (requires `qiskit-aer-gpu 0.13.4` + `cupy‑cuda12x`)


