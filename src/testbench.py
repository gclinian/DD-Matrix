#!/usr/bin/env python3

from dd_sparse import dd_sparse_numpy, GateType, GATE_UNITARIES, SingleQubitGate, ControlledGate
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


def test_simulation(num_qubits, num_gates, threshold=0.5):
    """
    Generate a random circuit and compare strong vs. weak simulation performance.
    """
    print(f"\nTesting {num_qubits} qubits, {num_gates} gates (threshold={threshold})")

    init_matrix_size = 2 ** num_qubits

    # Generate operations excluding ARBITRARY
    operations = random_operations(num_gates, num_qubits, threshold)

    # Strong simulation
    strong_result = dd_sparse_numpy(
        operations,
        init_matrix_size,
        num_qubits,
        MODE='strong_simulation',
        DEBUG=False
    )

    # Weak simulation
    weak_result = dd_sparse_numpy(
        operations,
        init_matrix_size,
        num_qubits,
        MODE='weak_simulation',
        DEBUG=False
    )

    # Display results
    print("Strong Simulation:")
    print(f"  Time Spent: {strong_result['time_spent']:.4f} s")
    print(f"  Avg CPU:    {strong_result.get('avg_cpu', 'N/A'):.1f}%")
    print(f"  Peak Mem:   {strong_result.get('peak_mem', 'N/A'):.1f} MB")
    if strong_result.get('error'):
        print(f"  Error:      {strong_result['error']}")

    print("Weak Simulation:")
    print(f"  Time Spent: {weak_result['time_spent']:.4f} s")
    print(f"  Avg CPU:    {weak_result.get('avg_cpu', 'N/A'):.1f}%")
    print(f"  Peak Mem:   {weak_result.get('peak_mem', 'N/A'):.1f} MB")
    if weak_result.get('error'):
        print(f"  Error:      {weak_result['error']}")

    print("" + "-" * 50)


def main():
    # Define test configurations: (num_qubits, num_gates)
    configs = [
        (3, 3)
    ]

    for nq, ng in configs:
        test_simulation(nq, ng)


if __name__ == "__main__":
    main()
