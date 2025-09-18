from src.mbdd.gates import *
from src.mbdd.circuit import *
from src.mbdd.simulator import *
from src.mbdd.comparator import *

import random
import time

def random_operations(num_operations, num_qubits, threshold=0.5):
    """
    Generate random gate operations excluding ARBITRARY gates.

    Args:
        num_operations (int): Number of operations to generate.
        num_qubits (int): Number of qubits.
        threshold (float): Probability of picking a single-qubit gate.

    Returns:
        List[GateOperation]: Generated operations list.
    """
    ops = []
    valid_gates = [g for g in GateType if g != GateType.ARBITRARY]
    while len(ops) < num_operations:
        gate = random.choice(valid_gates)
        matrix = GATE_UNITARIES[gate]
        a, b = matrix[0]
        c, d = matrix[1]

        if random.random() < threshold:
            target = random.randint(0, num_qubits - 1)
            ops.append(SingleQubitGate(gate, a, b, c, d, target))
        else:
            control, target = random.sample(range(num_qubits), 2)
            ops.append(ControlledGate(gate, a, b, c, d, target, [control]))
    return ops

def simulate_once(num_qubits, num_gates, threshold=0.5, operations=None):
    """
    Generate a random circuit and compare strong vs. weak simulation performance.
    """

    init_matrix_size = 2 * num_qubits + 1

    # Generate operations excluding ARBITRARY
    if operations is None:
        operations = random_operations(num_gates, num_qubits, threshold)
    
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

    # Compare results

    


def test_simulation():
    # Define test configurations: (num_qubits, num_gates)
    configs = [(3, 3) for i in range(1)]

    idx = 1
    for nq, ng in configs:
        print(f"Test {idx}/{len(configs)}: {nq} qubits, {ng} gates")
        idx += 1
        simulate_once(nq, ng)
