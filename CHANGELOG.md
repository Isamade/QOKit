# Changelog
All notable changes to **QOKit** are documented in this file.  
This project follows [Keep a Changelog](https://keepachangelog.com)
and semantic-ish version tags.


## [Unreleased] – 2025-07-05  *(statevector_gpu baseline, GPU batch optimiser,GPU async stream)))*
### Added
- - Batch L-BFGS-B now transfers the θ-matrix via pinned memory; host work
  overlaps GPU kernel (≈ 10–15 % faster than 0.4.2).- 
- **Async CUDA stream & pinned-host buffers**  
  `qokit/utils/cuda_stream.py` + automatic use in batch optimiser.
- **Batch L-BFGS-B** (`batch_minimize_lbfgsb`) and
  `Objective.batch_opt()` helper (.
- Smoke tests:  
  `test_async_stream_gpu`, `test_gpu_bench_smoke`, etc.

- **CuPy cost-vector kernel** `cost_vector_gpu()` .
- GPU parity unit-test (`test_cost_vector_gpu.py`).

- Backend selector `sim_backend.get_backend()`  
  – picks **`statevector_gpu`** when available .
- Initial GPU benchmark harness
  `scripts/benchmark_gpu_before_after.py`.
- ### Changed
- `get_qaoa_portfolio_objective()` auto-switches to CuPy kernel when
  `device="gpu"` **and** `precomputed_energies="vectorized"`.
- - Default optimiser for `scripts/run_sweep.py --device gpu`
  is now batch L-BFGS-B (`B=8`).

---

## [0.3.0] – 2025-07-01  *(CPU optimised release)*
### Added
- Vectorised objective, analytic gradients, diagonal phase-cache, …

### Changed
- Cost-vector evaluator now accepts batched input.

### Fixed
- Statevector reuse bug in batched objective.

