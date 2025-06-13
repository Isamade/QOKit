###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import qiskit
import numpy as np
import logging
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RXXGate, RYYGate  # Import if not already present

# Import local QOKit modules (ensure these paths are correct relative to qaoa_circuit_portfolio.py)
from .portfolio_optimization import yield_all_indices_cosntrained, get_configuration_cost
from .utils import reverse_array_index_bit_order, state_to_ampl_counts

# Setup logger for this module
module_logger = logging.getLogger(__name__)
if not module_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    module_logger.addHandler(handler)
    module_logger.setLevel(logging.INFO)  # Set to INFO or DEBUG for more details


def generate_dicke_state_fast(N, K):
    """
    Generate the dicke state with yield function (as a state vector).
    This is a fallback if qokit.dicke_state_utils.dicke_simple is not available.
    """
    index = yield_all_indices_cosntrained(N, K)
    s = np.zeros(2 ** N, dtype=complex)  # Ensure complex dtype for statevector
    for i in index:
        s[i] = 1
    # Normalize using L2 norm (sum of squares of magnitudes)
    s = s / np.sqrt(np.sum(np.abs(s) ** 2))
    return s


def get_cost_circuit(qr, J_coeffs, h_coeffs, gamma, T=1):
    """
    Construct the problem Hamiltonian layer for QAOA circuit.
    H = 0.5*q\sum_{i=1}^{n-1} \sum_{j=i+1}^n \sigma_{ij}Z_i Z_j + 0.5 \sum_i (-q\sum_{j=1}^n{\sigma_ij} + \mu_i) Z_i + Constant
    This function now takes J_coeffs and h_coeffs directly.
    """
    qc_cost = QuantumCircuit(qr)
    # Apply RZ gates for linear terms (h_i Z_i)
    for i in range(qr.size):
        # Use .get() with a default of 0 to handle cases where an index might not be in h_coeffs
        # Note: Qiskit's RZ expects angle directly, not 2*angle. So it should be just angle.
        # However, if your problem formulation includes a 0.5 or other factors, keep it consistent.
        # Assuming QOKit's convention expects 2 * coefficient * gamma * T for RZ and RZZ.
        if h_coeffs.get(i, 0) != 0:
            qc_cost.rz(2 * h_coeffs.get(i, 0) * gamma * T, qr[i])

    # Apply RZZ gates for quadratic terms (J_ij Z_i Z_j)
    for (i, j), val in J_coeffs.items():
        if val != 0:
            qc_cost.rzz(2 * val * gamma * T, qr[i], qr[j])
    return qc_cost


def get_dicke_init(N, K):
    """
    Generate dicke state in gates. Prioritizes qokit.dicke_state_utils.dicke_simple,
    falls back to generating from statevector if not available.
    """
    try:
        # Attempt to import the specific Dicke state utility from QOKit
        from qokit.dicke_state_utils import dicke_simple
        module_logger.info(f"Using dicke_simple from qokit.dicke_state_utils for N={N}, K={K}")
        return dicke_simple(N, K)
    except (ImportError, AttributeError) as e:
        # Fallback if dicke_simple is not found or cannot be imported
        module_logger.warning(f"Could not import or use qokit.dicke_state_utils.dicke_simple (Error: {e}). "
                              f"Falling back to initializing Dicke state from generated statevector for N={N}, K={K}.")
        dicke_sv = generate_dicke_state_fast(N, K)
        qc = QuantumCircuit(N)  # Create a quantum circuit with N qubits
        qc.initialize(dicke_sv, qc.qubits)  # Initialize with the generated statevector
        return qc


def get_mixer_Txy(qubits, beta, T=1, mixer_topology='complete'):
    """
    Generates the Trotterized XY Mixer Hamiltonian circuit (e^{-i beta H_M T}).
    """
    qc_mixer = QuantumCircuit(qubits)
    N = len(qubits)

    if mixer_topology == 'complete':
        for i in range(N):
            for j in range(i + 1, N):
                qc_mixer.rxx(2 * beta * T, qubits[i], qubits[j])
                qc_mixer.ryy(2 * beta * T, qubits[i], qubits[j])
    elif mixer_topology == 'linear':
        for i in range(N - 1):
            qc_mixer.rxx(2 * beta * T, qubits[i], qubits[i + 1])
            qc_mixer.ryy(2 * beta * T, qubits[i], qubits[i + 1])
    elif mixer_topology == 'ring':
        for i in range(N):
            j = (i + 1) % N
            qc_mixer.rxx(2 * beta * T, qubits[i], qubits[j])
            qc_mixer.ryy(2 * beta * T, qubits[i], qubits[j])
    else:
        raise ValueError(f"Unknown mixer_topology: {mixer_topology}. Choose from 'complete', 'linear', 'ring'.")

    return qc_mixer


def get_mixer_RX(qr, beta):
    """A layer of RX gates (transverse field mixer)."""
    N = qr.size
    qc = QuantumCircuit(qr)  # Create a new circuit for this mixer layer
    for i in range(N):
        qc.rx(2 * beta, qr[i])  # Apply RX gate
    return qc


def get_uniform_init(N):
    """
    Generates a uniform superposition initial state using Hadamard gates.
    """
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.h(i)
    return qc


# NOTE: This function get_qaoa_circuit seems to be an older utility within QOKit.
# Your current setup uses get_parameterized_qaoa_circuit.
# I've updated it for correctness if it were to be used, but it's not the main path here.
def get_qaoa_circuit(
        po_problem,
        gammas,
        betas,
        depth,
        ini="dicke",
        mixer="trotter_ring",
        T=1,
        ini_state=None,
        save_state=True,
        minus=False,  # Parameter 'minus' is not used in get_mixer_Txy and seems vestigial
        mixer_topology='complete'  # Added for compatibility if mixer_Txy is called
):
    """
    Put all ingredients together to build up a qaoa circuit
    """
    N = po_problem["N"]
    K = po_problem["K"]
    qr = QuantumRegister(N, 'q')  # Define QuantumRegister here for consistency

    circuit = QuantumCircuit(qr)  # Initialize the main circuit with the QuantumRegister

    if ini_state is not None:
        circuit.initialize(ini_state, qr)
    else:
        if ini.lower() == "dicke":
            initial_qc = get_dicke_init(N, K)
            circuit.compose(initial_qc, inplace=True)
        elif ini.lower() == "uniform":
            initial_qc = get_uniform_init(N)
            circuit.compose(initial_qc, inplace=True)
        else:
            raise ValueError("Undefined initial circuit")

    for i in range(depth):
        # Cost Hamiltonian
        circuit.compose(get_cost_circuit(qr, po_problem["J"], po_problem["h"], gammas[i], T=T), inplace=True)

        # Mixer Hamiltonian
        if mixer.lower() == "trotter_ring":
            circuit.compose(get_mixer_Txy(qr, betas[i], T=T, mixer_topology=mixer_topology), inplace=True)
        elif mixer.lower() == "rx":
            circuit.compose(get_mixer_RX(qr, betas[i]), inplace=True)
        else:
            raise ValueError("Undefined mixer circuit")

    if save_state is False:
        circuit.measure_all()
    return circuit


def get_parameterized_qaoa_circuit(po_problem, depth=1, ini_type='uniform', mixer_type='trotter_ring', T=1,
                                   simulator='qiskit', mixer_topology='complete', gamma_params=None, beta_params=None):
    """
    Returns the parameterized QAOA circuit for portfolio optimization.
    This is the main function called by get_qaoa_portfolio_objective.
    It now correctly handles parameter vectors for gamma and beta.
    """
    N = po_problem["N"]
    qr = QuantumRegister(N, 'q')
    qc_qaoa = QuantumCircuit(qr)

    # Use ParameterVector for gamma and beta if they are None (i.e. if it's the first call)
    # otherwise use the provided ParameterVector objects (for binding later)
    if gamma_params is None:
        gamma_params = ParameterVector('gamma', depth)
    if beta_params is None:
        beta_params = ParameterVector('beta', depth)

    if ini_type == 'uniform':
        for i in range(N):
            qc_qaoa.h(qr[i])
    elif ini_type == 'dicke':
        K = po_problem["K"]  # K is needed for Dicke state, get it from po_problem
        qc_initial = get_dicke_init(N, K=K)  # Call the robust get_dicke_init
        qc_qaoa.compose(qc_initial, inplace=True)
    else:
        raise ValueError(f"Unknown initial state type: {ini_type}")

    # QAOA layers
    for layer in range(depth):
        # Cost Hamiltonian (H_C)
        qc_qaoa.compose(get_cost_circuit(qr, po_problem["J"], po_problem["h"], gamma_params[layer], T=T), inplace=True)

        # Mixer Hamiltonian (H_M)
        if mixer_type == 'trotter_ring':  # This is the 'trotter_ring' type that you specified
            qc_qaoa.compose(get_mixer_Txy(qr, beta_params[layer], T=T, mixer_topology=mixer_topology), inplace=True)
        elif mixer_type == 'x_mixer':  # Corrected to call get_mixer_RX
            qc_qaoa.compose(get_mixer_RX(qr, beta_params[layer]), inplace=True)
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")

    return qc_qaoa, gamma_params, beta_params  # Return the circuit AND the parameter vectors


def get_energy_expectation(po_problem, samples):
    """Compute energy expectation from measurement samples"""
    expectation_value = 0
    N_total = 0
    for config, count in samples.items():
        expectation_value += count * get_configuration_cost(po_problem, config)
        N_total += count
    expectation_value = expectation_value / N_total

    return expectation_value


def get_energy_expectation_sv(po_problem, samples):
    """Compute energy expectation from full state vector"""
    expectation_value = 0
    # convert state vector to dictionary
    samples = state_to_ampl_counts(samples)
    for config, wf in samples.items():
        expectation_value += (np.abs(wf) ** 2) * get_configuration_cost(po_problem, config)

    return expectation_value


def invert_counts(counts):
    """convert qubit order for measurement samples (Qiskit returns little-endian)"""
    return {k[::-1]: v for k, v in counts.items()}


def measure_circuit(circuit, n_trials=1024, save_state=True):
    """Get the output from circuit, either measured samples or full state vector"""
    if save_state is False:
        backend = Aer.get_backend("qasm_simulator")
        job = transpile(circuit, backend, shots=n_trials)
        result = job.result()
        bitstrings = invert_counts(result.get_counts())
        return bitstrings
    else:
        backend = Aer.get_backend("statevector_simulator")
        circ = transpile(circuit, backend)
        state = Statevector(circ)
        return reverse_array_index_bit_order(state)


# This function is used by get_qaoa_portfolio_objective to create the objective function for the optimizer.
# It uses the parameterized circuit and binds the values for optimization.
def circuit_measurement_function(
        po_problem,
        p,
        ini="dicke",
        mixer="trotter_ring",
        T=None,
        ini_state=None,
        n_trials=1024,
        save_state=True,
        minus=False,
        mixer_topology='complete'
):
    """Helper function to define the objective function to optimize"""

    # Create the parameterized QAOA circuit (which returns the circuit and parameter vectors)
    parameterized_circuit, gammas_params, betas_params = get_parameterized_qaoa_circuit(
        po_problem=po_problem,
        depth=p,
        ini_type=ini,
        mixer_type=mixer,
        T=T,
        # simulator is not used in this function, but in the higher-level objective
        mixer_topology=mixer_topology
    )

    def f(x):
        # 'x' contains the numerical values for gammas and betas from the classical optimizer
        gammas_val = x[0:p]
        betas_val = x[p:]

        # Bind the numerical values from the optimizer to the ParameterVector objects
        bound_circuit = parameterized_circuit.assign_parameters({
            gammas_params[i]: gammas_val[i] for i in range(p)
        })
        bound_circuit = bound_circuit.assign_parameters({
            betas_params[i]: betas_val[i] for i in range(p)
        })

        samples = measure_circuit(bound_circuit, n_trials=n_trials, save_state=save_state)
        if save_state is False:
            energy_expectation_value = get_energy_expectation(po_problem, samples)
        else:
            energy_expectation_value = get_energy_expectation_sv(po_problem, samples)
        return energy_expectation_value

    return f