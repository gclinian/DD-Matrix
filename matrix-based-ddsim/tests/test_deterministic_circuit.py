from src.mbdd.gates import *
from src.mbdd.circuit import *
from src.mbdd.simulator import *

import random
import time


def simulate_once(num_qubits, num_gates, threshold=0.5, operations=None):
    """
    Generate a random circuit and compare strong vs. weak simulation performance.
    """

    init_matrix_size = 2 * num_qubits + 1

    # Generate operations
    operations : List[GateOperation] = []
    operations.append(operation_constructor(GateType.H, 3))
    operations.append(operation_constructor(GateType.H, 2))
    operations.append(operation_constructor(GateType.H, 0))
    operations.append(operation_constructor(GateType.X, 1, [0, 2, 3]))
    
    view_operations(operations)

    # Weak simulation
    weak_result = dd_sparse_numpy(
        operations,
        init_matrix_size,
        num_qubits,
        MODE='weak_simulation'
    )

    print("Weak Simulation:")
    print(f"  Time Spent: {weak_result['time_spent']:.4f} s")
    print(f"  Avg CPU:    {weak_result.get('avg_cpu', 'N/A'):.1f}%")
    print(f"  Peak Mem:   {weak_result.get('peak_mem', 'N/A'):.1f} MB")
    if weak_result.get('error'):
        print(f"  Error:      {weak_result['error']}")

    print("" + "-" * 30)

    
    # Strong simulation
    strong_result = dd_sparse_numpy(
        operations,
        init_matrix_size,
        num_qubits,
        MODE='strong_simulation'
    )
    
    # Display results
    print("Strong Simulation:")
    print(f"  Time Spent: {strong_result['time_spent']:.4f} s")
    print(f"  Avg CPU:    {strong_result.get('avg_cpu', 'N/A'):.1f}%")
    print(f"  Peak Mem:   {strong_result.get('peak_mem', 'N/A'):.1f} MB")
    if strong_result.get('error'):
        print(f"  Error:      {strong_result['error']}")
    
    print("" + "-" * 50)

def test_simulation():
    # Define test configurations: (num_qubits, num_gates)
    configs = [(4, 4) for i in range(1)]

    idx = 1
    for nq, ng in configs:
        print(f"Test {idx}/{len(configs)}: {nq} qubits, {ng} gates")
        idx += 1
        simulate_once(nq, ng)
