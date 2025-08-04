#!/usr/bin/env python3

from dd_sparse import *
import random
import time
from sys import argv


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

def _view_operations(operations: List[GateOperation]):
    for op in operations:
        print(op.gate.name, op.target)

def test_simulation(num_qubits, num_gates, threshold=1):
    """
    Generate a random circuit and compare strong vs. weak simulation performance.
    """

    init_matrix_size = 2 * num_qubits + 1

    # Generate operations excluding ARBITRARY
    operations = random_operations(num_gates, num_qubits, threshold)
    
    # _view_operations(operations)

    

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

    


def main(num_qubits, num_gates):
    # Define test configurations: (num_qubits, num_gates)
    configs = [(num_qubits, num_gates) for i in range(20)]

    idx = 1
    for nq, ng in configs:
        print(f"Test {idx}/{len(configs)}: {nq} qubits, {ng} gates")
        idx += 1
        test_simulation(nq, ng)


if __name__ == "__main__":
    main(int(argv[1]), int(argv[2]))
