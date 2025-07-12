# std / qiskit
from qiskit_aer import AerSimulator
import numpy as np
# project imports
from qokit.fur.qaoa_simulator_base import QAOAFastSimulatorBase
# ▲ real circuit builder that already exists in your code-base
from qokit.qaoa_circuit_portfolio import get_parameterized_qaoa_circuit

class FastCPUSimulator(QAOAFastSimulatorBase):
    """
    Fast pure-CPU simulator used for the Pauli-X mixer.

    Parameters
    ----------
    N : int
        Number of qubits.
    costs, terms : see original args.
    backend : qiskit_aer.AerSimulator | None
        Optional explicit backend.  If provided (e.g. statevector_gpu)
        every call to `backend.run()` will use that instance instead of
        the default CPU one.
    """

    def __init__(self,N, *, costs, terms, backend=None, **_):
        self.mixer = "rx"
        super().__init__(n_qubits=N, costs=np.asarray(costs), terms=terms, backend=backend or AerSimulator(method="statevector"))

    # ---------- public API identical to original class ---------------- #
    def get_cost_diagonal(self):
        return np.asarray(self.costs)

    def simulate_qaoa(self, gamma, beta, initial_state, n_trotters=1):
        """
               Run QAOA circuit on the stored backend (CPU or GPU).
               gamma, beta may be 1-D numpy arrays of length p.
               """
        backend = self.backend or AerSimulator(method="statevector")

        # ---------------------------------------------------------------
        # Build a MINIMAL portfolio-dict so the helper works
        # ---------------------------------------------------------------
        po_stub = {
            "N": self.n_qubits,
            "K": 0,
            "q": 0.7,  # any q is fine for circuit
            "means": np.zeros(self.n_qubits),
            "cov": np.eye(self.n_qubits),
        }
        if self.mixer.lower() != "rx":
            raise ValueError("FastCPUSimulator supports mixer='rx' only")

        qc = get_parameterized_qaoa_circuit(
            po_stub,
            depth=len(gamma),
            ini="dicke",
            mixer="rx",
            T=n_trotters,
        ).assign_parameters(np.hstack([beta, gamma]))

        if not any(instr.operation.name == "save_statevector" for instr in qc.data):
            qc.save_statevector()

        sv = backend.run(qc).result().get_statevector()
        return np.asarray(sv)

    def get_expectation(self, sv, costs, *, preserve_state, optimization_type):
        probs = np.abs(sv) ** 2
        energy = costs.dot(probs)
        return -energy if optimization_type == "max" else energy

    def get_overlap(self, sv, costs, indices, *, preserve_state, optimization_type):
        probs = np.abs(sv) ** 2
        return probs[indices].sum()
    def _diag_from_costs(self, costs):
        return np.asarray(costs)

    def _diag_from_terms(self, terms):
        return np.asarray(terms)

    def get_probabilities(self, sv):
        return np.abs(sv) ** 2

    def get_statevector(self, circuit):
        return np.asarray(self.backend.run(circuit).result().get_statevector())
