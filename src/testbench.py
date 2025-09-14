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
    print("[Operations]")
    for op in operations:
        if isinstance(op, SingleQubitGate):
            print(op.gate.name, op.target)
        elif isinstance(op, ControlledGate):
            print(op.gate.name, op.target, "Control:", op.control)
    print("-" * 30)

def test_simulation(num_qubits, num_gates, threshold=0.5, operations=None):
    """
    Generate a random circuit and compare strong vs. weak simulation performance.
    """

    init_matrix_size = 2 * num_qubits + 1

    # Generate operations excluding ARBITRARY
    if operations is None:
        operations = random_operations(num_gates, num_qubits, threshold)
    
    # SDG 4
    # H 0 Control: [3]
    # Y 2 Control: [3]
    # SX 1 Control: [0]
    # TDG 4 Control: [1]
    # H 2 Control: [0]
    # X 4 Control: [2]
    # Y 4 Control: [3]
    # H 2 Control: [0]
    # T 1
    
    # operations = [
    #                 # operation_constructor(GateType.SDG, 4), 
    #                 # operation_constructor(GateType.H, 0, [3]),
    #                 # operation_constructor(GateType.Y, 2, [3]),
    #                 # operation_constructor(GateType.SX, 1, [0]),
    #                 operation_constructor(GateType.TDG, 4, [1]),
    #                 operation_constructor(GateType.H, 2, [0]),
    #                 operation_constructor(GateType.X, 4, [2]),
    #                 operation_constructor(GateType.Y, 4, [3]),
    #                 operation_constructor(GateType.X, 2, [0]),
    #                 # operation_constructor(GateType.T, 1)
    #              ]
    

    
    _view_operations(operations)

    

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
    # strong_result = dd_sparse_numpy(
    #     operations,
    #     init_matrix_size,
    #     num_qubits,
    #     MODE='strong_simulation'
    # )
    
    # Display results
    # print("Strong Simulation:")
    # print(f"  Time Spent: {strong_result['time_spent']:.4f} s")
    # print(f"  Avg CPU:    {strong_result.get('avg_cpu', 'N/A'):.1f}%")
    # print(f"  Peak Mem:   {strong_result.get('peak_mem', 'N/A'):.1f} MB")
    # if strong_result.get('error'):
    #     print(f"  Error:      {strong_result['error']}")
    
    # print("" + "-" * 50)

    


def main(num_qubits, num_gates):
    # Define test configurations: (num_qubits, num_gates)
    configs = [(num_qubits, num_gates) for i in range(1)]

    idx = 1
    for nq, ng in configs:
        print(f"Test {idx}/{len(configs)}: {nq} qubits, {ng} gates")
        idx += 1
        test_simulation(nq, ng)


if __name__ == "__main__":
    main(int(argv[1]), int(argv[2]))
