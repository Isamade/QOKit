# QOKit/sim_backend.py
"""
Return a Qiskit AerSimulator configured for CPU or GPU.

Usage:
    backend = get_backend("gpu")   # statevector_gpu
    backend = get_backend()        # default → cpu
"""

from qiskit_aer import AerSimulator, AerError


def get_backend(device="cpu"):
    if device.lower() == "gpu":
        try:
            return AerSimulator(method="statevector_gpu")
        except AerError:
            # GPU build not installed – warn & fall back
            import warnings
            warnings.warn("GPU backend not available; falling back to CPU.")
    return AerSimulator(method="statevector")