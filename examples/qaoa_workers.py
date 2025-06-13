import os
import logging
import time
import numpy as np
# from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator

# Import QOKit specific modules
# Make sure your QOKit path is correctly set in your environment or sys.path in the notebook
# The notebook's sys.path modification propagates to workers for imports
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from qokit.utils import reverse_array_index_bit_order, state_to_ampl_counts

# Worker-specific logger setup
worker_logger = logging.getLogger(f"SpawnPoolWorker-{os.getpid()}")
# This global setup ensures the handler is added only once per worker process
# The level is set by the initializer function.
if not worker_logger.handlers:
    worker_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    worker_handler.setFormatter(formatter)
    worker_logger.addHandler(worker_handler)
    worker_logger.propagate = False # Prevent messages from bubbling up to the root logger

# This function is called by the multiprocessing pool to initialize each worker process.
# It's used here to set the logging level for the worker's logger.
def worker_init(logging_level):
    worker_logger.setLevel(logging_level)
    worker_logger.info(f"Worker {os.getpid()}: Initialized with logging level {logging.getLevelName(logging_level)}.")
    # Set environment variables for the worker process
    # These are picked up from os.environ when the worker process starts
    # We log them here to confirm they are set within the worker's context
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        worker_logger.debug(f"Worker {os.getpid()}: ENV {var}={os.environ.get(var)}")


def run_single_optimization_optimized(
    run_id,
    po_problem_data,
    p_value,
    initial_point_override,
    ini_type,
    mixer_type,
    T_value,
    simulator_type,
    num_shots,
    simulator_backend_obj,
    evals_per_point,
    optimizer_method,
    optimizer_options,
    env_vars, # This parameter is informative, not directly used to set env vars here
    max_iterations, # This parameter is informative, optimizer_options should contain maxiter
    mixer_topology,
    K_value # This is now explicitly passed for Dicke state
):
    """
    Performs a single QAOA optimization run within a worker process.
    """
    worker_logger.info(f"Worker {os.getpid()} (Run ID: {run_id}) starting optimization...")

    # Set environment variables for this specific worker process for this run
    # Note: For 'spawn' method, these need to be set before pool creation if they affect
    # module imports or global settings. However, setting them here within the worker
    # ensures they apply to the libraries used by this specific worker process.
    # The main script sets them globally, but individual worker re-setting is good practice.
    for var, val in env_vars.items():
        os.environ[var] = str(val)
        worker_logger.debug(f"Worker {os.getpid()}: Set {var}={os.environ.get(var)}")

    # Define a callback for the optimizer to provide updates
    def optimizer_progress_callback(xk):
        # xk contains the current parameters from the optimizer
        worker_logger.info(f"Worker {os.getpid()} (Run ID: {run_id}): Optimizer iteration completed with current parameters.")
        # You could add more detailed logging here, like current energy,
        # but that would require re-evaluating the objective, which can be slow.

    try:
        # Create the QAOA objective function
        # This calls get_parameterized_qaoa_circuit internally
        qaoa_obj = get_qaoa_portfolio_objective(
            po_problem=po_problem_data,
            p=p_value,
            ini=ini_type,
            mixer=mixer_type,
            T=T_value,
            simulator=simulator_backend_obj if simulator_type == 'qiskit' else simulator_type,
            mixer_topology=mixer_topology,
            K=K_value # Pass K here for Dicke state initialization
        )

        # Handle initial point if needed (e.g., convert initial_point_override to appropriate format)
        # For 'dicke', QOKit handles it internally, no initial_point_override usually.
        initial_point_for_optimizer = None
        if initial_point_override is not None:
            # Handle conversion or specific initial point logic if 'initial_point_override' is used
            # For now, we assume it's None and QOKit handles ini_type='dicke'
            pass

        optimizer_start_time = time.perf_counter()
        res = qaoa_obj.optimize(
            initial_point=initial_point_for_optimizer,
            method=optimizer_method,
            options=optimizer_options,
            # maxiter is passed via optimizer_options, but sometimes QOKit might override it or use directly
            # Ensure max_iterations is correctly mapped into optimizer_options
            callback=optimizer_progress_callback # Pass the callback function
        )
        optimizer_end_time = time.perf_counter()
        optimizer_time = optimizer_end_time - optimizer_start_time

        # Extract results
        success = res.success if hasattr(res, 'success') else (res.status == 0) # Common scipy.optimize success check
        energy = res.fun if hasattr(res, 'fun') else float('inf') # Final energy value
        nfev = res.nfev if hasattr(res, 'nfev') else -1 # Number of function evaluations

        worker_logger.info(f"Worker {os.getpid()} (Run ID: {run_id}): Optimization completed. Success: {success}, Energy: {energy:.6f}, NFEV: {nfev}, Opt Time: {optimizer_time:.2f}s")

        return {
            "success": success,
            "energy": energy,
            "nfev": nfev,
            "optimizer_time": optimizer_time,
            "run_id": run_id,
            "initial_point_strategy": ini_type,
            "optimizer_method": optimizer_method,
            "N_qubits": po_problem_data["N"],
            "p_value": p_value,
            "env_vars": env_vars # Return the environment variables for this run
        }

    except Exception as e:
        worker_logger.error(f"Worker {os.getpid()} (Run ID: {run_id}): Optimization failed with an error: {e}", exc_info=True)
        return {
            "success": False,
            "energy": float('inf'),
            "nfev": -1,
            "optimizer_time": -1.0,
            "run_id": run_id,
            "initial_point_strategy": ini_type,
            "optimizer_method": optimizer_method,
            "N_qubits": po_problem_data["N"],
            "p_value": p_value,
            "env_vars": env_vars
        }