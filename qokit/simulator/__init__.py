from .fast_cpu import FastCPUSimulator
from .qiskit_cpu import QiskitCPUSimulator

SIMULATORS = {
    "x": {
        "fast": FastCPUSimulator,
        "qiskit": QiskitCPUSimulator,
    },
    "xy": {
        "qiskit": QiskitCPUSimulator,
    },
}
