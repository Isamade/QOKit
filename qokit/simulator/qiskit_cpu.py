import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qokit.qaoa_circuit_portfolio import get_parameterized_qaoa_circuit
from qokit.utils import reverse_array_index_bit_order
from qokit.fur.qaoa_simulator_base import QAOAFastSimulatorBase


class QiskitCPUSimulator(QAOAFastSimulatorBase):
    """
    Aer wrapper that works for **both** mixers:

        • mixer="rx"            (Pauli-X)
        • mixer="trotter_ring"  (nearest-neighbour XY)

    When you inject backend=statevector_gpu it runs entirely on CUDA.
    """

    def __init__(self, N, *, costs, terms, backend=None, **_):
        super().__init__(N, costs=costs, terms=terms,
                         backend=backend or AerSimulator(method="statevector"))
        self.costs = np.asarray(costs)

    # ------------------------------------------------------------------
    def simulate_qaoa(
        self,
        gamma,
        beta,
        initial_state=None,
        *,
        n_trotters: int = 1,
        mixer: str = "rx",
    ):
        mixer = mixer.lower()
        if mixer not in ("rx", "trotter_ring"):
            raise ValueError("mixer must be 'rx' or 'trotter_ring'")

        qc = get_parameterized_qaoa_circuit(
            {"N": self.n_qubits, "K": 0, "q": 0.7,
             "means": np.zeros(self.n_qubits), "cov": np.eye(self.n_qubits)},
            depth=len(gamma),
            ini="dicke",
            mixer=mixer,
            T=n_trotters,
            save_state=False,        # we’ll add statevector manually
        ).assign_parameters(np.hstack([beta, gamma]))

        qc.save_statevector()
        circ = transpile(qc, self.backend)
        sv = reverse_array_index_bit_order(Statevector(circ))
        return np.asarray(sv)

    # ------------------------------------------------------------------
    def get_cost_diagonal(self):         return self.costs
    def get_probabilities(self, sv):     return np.abs(sv)**2
    def get_statevector(self, circ):     return np.asarray(
        self.backend.run(circ).result().get_statevector())

    def get_expectation(self, sv, costs, *, preserve_state, optimization_type):
        probs  = np.abs(sv)**2
        e = float(costs.dot(probs))
        return -e if optimization_type == "max" else e

    def get_overlap(self, sv, costs, indices, *, preserve_state, optimization_type):
        return float(np.abs(sv)**2 [indices].sum())
