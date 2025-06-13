# ~/PycharmProjects/QOKit/examples/run_qaoa_test.py

import os
import json
import numpy as np
import time
import logging
from multiprocessing import Pool, current_process
from datetime import datetime
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# Qiskit and QOKit imports that might be needed by helper functions,
# though primarily used in the main notebook.
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QAOAAnsatz
from qokit.qaoa_circuit_portfolio import get_parameterized_qaoa_circuit
from qokit.portfolio_optimization import portfolio_brute_force # For classical solution if N <= 20

# --- Helper function for multiprocessing workers ---
def worker_init(log_level):
    """
    Initializes each worker process for multiprocessing.
    Sets the logging level and names the process.
    """
    logging.basicConfig(level=log_level, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    current_process().name = f"Worker-{current_process().pid}"
    logging.info(f"Worker process {current_process().name} initialized with log level {logging.getLevelName(log_level)}")

# IMPORTANT: No other global-scope execution code should be in this file.
# All problem definition, QAOA circuit setup, and benchmarking loops
# must be in your main Jupyter Notebook or a dedicated execution script.