import time
import psutil
import os
import threading
import traceback
import random
import copy
from pynvml import *
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import cmath
import math
import re
from pathlib import Path

class GateType(Enum):
    X = auto()
    Y = auto()
    Z = auto()
    H = auto()
    S = auto()
    SDG = auto()
    T = auto()
    TDG = auto()
    SX = auto()
    V = auto()
    V_plus = auto()
    ARBITRARY = auto()

GATE_UNITARIES: Dict[GateType, List[List[complex]]]= {
    GateType.X: [[0, 1],
                [1, 0]],
    GateType.Y: [[0, -1j],
                [1j, 0]],
    GateType.Z: [[1, 0],
                [0, -1]],
    GateType.H: [[1/2**0.5, 1/2**0.5],
                [1/2**0.5, -1/2**0.5]],
    GateType.S:   [[1, 0], 
                   [0,  1j]],
    GateType.SDG: [[1, 0], 
                   [0, -1j]],
    GateType.T:   [[1, 0], 
                   [0,  cmath.exp(1j*math.pi/4)]],
    GateType.TDG: [[1, 0], 
                   [0,  cmath.exp(-1j*math.pi/4)]],
    GateType.SX:  [[0.5+0.5j, 0.5-0.5j],
                   [0.5-0.5j, 0.5+0.5j]],
    GateType.V:   [[0.5+0.5j, 0.5-0.5j], 
                  [0.5-0.5j, 0.5+0.5j]],
    GateType.V_plus:  [[0.5-0.5j, 0.5+0.5j], 
                  [0.5+0.5j, 0.5-0.5j]],
    GateType.ARBITRARY: [[1, 0], 
                         [0, 1]]  # Identity as a placeholder for arbitrary gates
}

@dataclass
class SingleQubitGate:
    gate: GateType
    a: complex
    b: complex
    c: complex
    d: complex
    target: int

@dataclass
class ControlledGate:
    gate: GateType
    a: complex
    b: complex
    c: complex
    d: complex
    target: int
    control: List[int]

GateOperation = SingleQubitGate | ControlledGate

def operation_constructor(gate: GateType, 
                          target: int, 
                          control: Optional[List[int]] = None, 
                          mode: Optional[str] = None,
                          matrix: Optional[List[List[complex]]] = None,
                          angles: Optional[List[float]] = None
) -> GateOperation:
    
    """
    Construct a quantum gate operation, either from a built-in gate type or an arbitrary unitary.

    This function returns either a SingleQubitGate or a ControlledGate depending on whether
    control qubits are specified. For GateType.ARBITRARY, you must specify `mode` as either
    'angles' (to generate the U3 matrix from Euler angles) or 'matrix' (to supply your own
    2x2 unitary).

    Args:
        gate (GateType): 
            The gate type to construct. Use GateType.ARBITRARY for custom unitaries.
        target (int): 
            Index of the target qubit.
        control (Optional[List[int]]): 
            List of control-qubit indices. If None, a single-qubit gate is returned.
        mode (Optional[str]): 
            Only for GateType.ARBITRARY.  
            - 'angles': compute the unitary from `angles`.  
            - 'matrix': validate and use the provided `matrix`.
        matrix (Optional[List[List[complex]]]): 
            A 2x2 complex matrix for ARBITRARY gates when `mode='matrix'`.
        angles (Optional[List[float]]): 
            Three Euler angles [theta, phi, lambda] for ARBITRARY gates when `mode='angles'`.

    Returns:
        GateOperation: 
            A SingleQubitGate or ControlledGate instance encapsulating the specified operation.

    Raises:
        ValueError: 
            - If `mode` is missing or invalid for an ARBITRARY gate.  
            - If `angles` is not a list of exactly three floats when `mode='angles'`.  
            - If `matrix` is missing, not 2x2, or not unitary when `mode='matrix'`.
    """

    def is_unitary(matrix: List[List[complex]], tol=1e-8) -> bool:
        M = np.array(matrix, dtype=complex)
        return np.allclose(M.conj().T @ M, np.eye(2), atol=tol)
    
    def check_matrix(matrix: List[List[complex]]):
        if matrix is None:
            raise ValueError("ARBITRARY gate requires a 2x2 matrix.")
        if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
            raise ValueError("Matrix must be 2x2 for arbitrary gate.")
        if not is_unitary(matrix):
            raise ValueError("Provided matrix is not unitary.")
    
    def handle_angles(angles: List[float]) -> List[List[complex]]:
        if angles is None or len(angles) != 3:
            raise ValueError("Angles must be provided as a list of three values (theta, phi, lambda).")
        theta, phi, lam = angles
        a = np.cos(theta / 2)
        b = -np.exp(1j * lam) * np.sin(theta / 2)
        c = np.exp(1j * phi) * np.sin(theta / 2)
        d = np.exp(1j * (phi + lam)) * np.cos(theta / 2)
        matrix = [[a, b], [c, d]]
        return matrix

    if gate is GateType.ARBITRARY:
        if mode is None:
            raise ValueError("Mode must be specified for ARBITRARY gate.")
        if mode == 'angles':
            matrix = handle_angles(angles)
        elif mode == 'matrix':
            check_matrix(matrix)
        else:
            raise ValueError("Invalid mode for ARBITRARY gate. Use 'angles' or 'matrix'.")
    else:
        matrix = GATE_UNITARIES[gate]
    
    a, b = matrix[0]
    c, d = matrix[1]
    
    if control is None:
        return SingleQubitGate(gate, a, b, c, d, target)
    else:
        return ControlledGate(gate, a, b, c, d, target, control)

def randomizer(num_qubits: int, num_operations: int, option: Optional[GateOperation], threshold=0) -> List[GateOperation]:
    operations: List[GateOperation] = []

    for _ in range(num_operations):
        gate = random.choice(list(GateType))
        matrix = GATE_UNITARIES[gate]
        a, b = matrix[0]
        c, d = matrix[1]
        
        if option == SingleQubitGate or random.random() < threshold:
            target = random.randint(0, num_qubits - 1)
            operations.append(SingleQubitGate(gate, a, b, c, d, target))
        elif option == ControlledGate or random.random() >= threshold:
            control, target = random.sample(range(num_qubits), 2)
            operations.append(ControlledGate(gate, a, b, c, d, target, [control]))
    
    return operations

def initialization(num_qubits: int, init_matrix_size: int, L, R) -> Tuple[List[List[complex]], List[List[complex]], List[List[int]]]:    
    if 1 < init_matrix_size:
        L[1, 0] = 1  

    for col in range(1, init_matrix_size - 2):  
        row = col + 2
        if row < init_matrix_size:
            L[row, col] = 1

    if 2 < init_matrix_size:
        R[2, 0] = 1  

    for col in range(1, init_matrix_size - 2):  
        if col % 2 == 1:  
            row = col + 3
        else:  
            row = col + 2
        if row < init_matrix_size:
            R[row, col] = 1

    Mapping = []
    Mapping.append([0])

    for i in range(1, num_qubits):
        Mapping.append([2*i - 1, 2*i])
    Mapping.append([2*i + 1])
    Mapping.append([2*i + 2])

    return L, R, Mapping

def initialization_sparse(INIT_MATRIX_SIZE, NUM_QUBITS, TYPE):
    if TYPE == 'numpy':
        import numpy as cp
        from scipy import sparse
    elif TYPE == 'cupy':
        import cupy as cp
        from cupyx.scipy import sparse
        
    row_L, col_L, data_L = [], [], []
    row_R, col_R, data_R = [], [], []

    if 1 < INIT_MATRIX_SIZE:
        row_L.append(1)
        col_L.append(0)
        data_L.append(1)

    for col in range(1, INIT_MATRIX_SIZE - 2):
        row = col + 2
        if row < INIT_MATRIX_SIZE:
            row_L.append(row)
            col_L.append(col)
            data_L.append(1)

    if 2 < INIT_MATRIX_SIZE:
        row_R.append(2)
        col_R.append(0)
        data_R.append(1)

    for col in range(1, INIT_MATRIX_SIZE - 2):
        row = col + 3 if col % 2 == 1 else col + 2
        if row < INIT_MATRIX_SIZE:
            row_R.append(row)
            col_R.append(col)
            data_R.append(1)

    L = sparse.csr_matrix((cp.array(data_L, dtype=cp.complex128), (cp.array(row_L), cp.array(col_L))),
                          shape=(INIT_MATRIX_SIZE, INIT_MATRIX_SIZE))

    R = sparse.csr_matrix((cp.array(data_R, dtype=cp.complex128), (cp.array(row_R), cp.array(col_R))),
                          shape=(INIT_MATRIX_SIZE, INIT_MATRIX_SIZE))
    
    def ones_positions_k(k: int):
        # [0], then pairs (1,2), (3,4), ..., (2k-3, 2k-2), then singles [2k-1], [2k]
        pos = [[0]]
        pos += [[i, i+1] for i in range(1, 2*k-1, 2)]  # k-1 pairs
        pos += [[2*k-1], [2*k]]
        return pos

    def wide_diag_matrix(k: int) -> sparse.csr_matrix:
        positions = ones_positions_k(k)
        rows = len(positions)          # k + 2
        cols = 2*k + 1                 # columns 0..2k

        # Build CSR arrays directly (indptr/indices/data)
        indptr = [0]
        indices = []
        data = []
        for idxs in positions:
            indices.extend(idxs)
            data.extend([1]*len(idxs))
            indptr.append(len(indices))

        indptr = cp.array(indptr, dtype=np.int32)
        indices = cp.array(indices, dtype=np.int32)
        data = cp.ones_like(indices, dtype=np.float32)  # or np.int32 if you prefer

        return sparse.csr_matrix((data, indices, indptr), shape=(rows, cols))
    # Mapping = [[0]]
    # for i in range(1, NUM_QUBITS):
    #     Mapping.append([2*i - 1, 2*i])
    # Mapping.append([2*NUM_QUBITS - 1])
    # Mapping.append([2*NUM_QUBITS])

    Mapping = wide_diag_matrix(NUM_QUBITS)

    return L, R, Mapping

def view_operations(operations: List[GateOperation]):
    print("[Operations]")
    for op in operations:
        if isinstance(op, SingleQubitGate):
            print(op.gate.name, op.target)
        elif isinstance(op, ControlledGate):
            print(op.gate.name, op.target, "Control:", op.control)
    print("-" * 30)

