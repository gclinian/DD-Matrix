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

import inspect

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


def randomizer(NUM_OPERATIONS: int, NUM_QUBITS: int, option: Optional[GateOperation], threshold=0) -> List[GateOperation]:
    operations: List[GateOperation] = []

    for _ in range(NUM_OPERATIONS):
        gate = random.choice(list(GateType))
        matrix = GATE_UNITARIES[gate]
        a, b = matrix[0]
        c, d = matrix[1]

        if option == SingleQubitGate or random.random() < threshold:
            target = random.randint(0, NUM_QUBITS - 1)
            operations.append(SingleQubitGate(gate, a, b, c, d, target))
        elif option == ControlledGate or random.random() >= threshold:
            control, target = random.sample(range(NUM_QUBITS), 2)
            operations.append(ControlledGate(gate, a, b, c, d, target, [control]))

    return operations

def initialization(L, R, NUM_QUBITS: int, INIT_MATRIX_SIZE: int) -> Tuple[List[List[complex]], List[List[complex]], List[List[int]]]:    
    if 1 < INIT_MATRIX_SIZE:
        L[1, 0] = 1  

    for col in range(1, INIT_MATRIX_SIZE - 2):  
        row = col + 2
        if row < INIT_MATRIX_SIZE:
            L[row, col] = 1

    if 2 < INIT_MATRIX_SIZE:
        R[2, 0] = 1  

    for col in range(1, INIT_MATRIX_SIZE - 2):  
        if col % 2 == 1:  
            row = col + 3
        else:  
            row = col + 2
        if row < INIT_MATRIX_SIZE:
            R[row, col] = 1

    Mapping = []
    Mapping.append([0])

    for i in range(1, NUM_QUBITS):
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

    Mapping = [[0]]
    for i in range(1, NUM_QUBITS):
        Mapping.append([2*i - 1, 2*i])
    Mapping.append([2*NUM_QUBITS - 1])
    Mapping.append([2*NUM_QUBITS])

    return L, R, Mapping

def _view_operations(operations: List[GateOperation]):
    for op in operations:
        print(op.gate.name, op.target)

def compute_result(L, R, seq: List[int], leaf: List[int], TYPE) -> complex:
    if TYPE == 'numpy':
        import numpy as cp
        from scipy import sparse
    elif TYPE == 'cupy':
        import cupy as cp
        from cupyx.scipy import sparse

    A = cp.zeros((1, L.shape[0]))

    A[0, leaf] = 1

    B = cp.zeros((L.shape[0], 1))
    B[0, 0] = 1

    dirs = B
    for dir in seq:
        if dir == 0:
            dirs = L @ dirs
        else:
            dirs = R @ dirs

    result = A @ dirs

    return result.item()

class SimulationAllPath:
    def __init__(self, L, R, leaf, NUM_QUBITS, TYPE, DEBUG=True):
        self.L = L
        self.R = R
        self.LL = self.L@self.L
        self.RR = self.R@self.R
        self.LR = self.L@self.R
        self.RL = self.R@self.L
        self.leaf = leaf
        self.NUM_QUBITS = NUM_QUBITS
        self.TYPE = TYPE
        self.DEBUG = DEBUG
        self.sim_result = {}

    def recursion(self, seq: List[int], dirs, A, start_time):
        if time.time() - start_time > 60:
            raise TimeoutError("Simulation Timeout: 60s.")
        if len(seq) == self.NUM_QUBITS:
            result = A @ dirs
            val = result.toarray()[0, 0]
            if abs(val) > 1e-10:
                self.sim_result[tuple(seq)] = val
            return

        if self.NUM_QUBITS <= 10 or len(seq) < 2:
            for dir in [0, 1]:
                next_dirs = self.L @ dirs if dir == 0 else self.R @ dirs
                self.recursion(seq + [dir], next_dirs, A, start_time)
        else:
            for d1 in [0, 1]:
                for d2 in [0, 1]:
                    if d1 == 0 and d2 == 0:
                        next_dirs = self.LL @ dirs
                    elif d1 == 0 and d2 == 1:
                        next_dirs = self.RL @ dirs
                    elif d1 == 1 and d2 == 0:
                        next_dirs = self.LR @ dirs
                    elif d1 == 1 and d2 == 1:
                        next_dirs = self.RR @ dirs

                    self.recursion(seq + [d1, d2], next_dirs, A, start_time)

    def simulate_non_recursive(self, A, B, start_time):
        stack = [([], B)]  # (sequence_so_far, current_dirs)

        while stack:
            if time.time() - start_time > 60:
                raise TimeoutError("Simulation Timeout: 60s.")

            seq, dirs = stack.pop()

            if len(seq) == self.NUM_QUBITS:
                result = A @ dirs
                val = result.toarray()[0, 0]
                if abs(val) > 1e-10:
                    self.sim_result[tuple(seq)] = val
                continue

            if self.NUM_QUBITS <= 10 or len(seq) == self.NUM_QUBITS - 1:
                for dir in [0, 1]:
                    next_dirs = self.L @ dirs if dir == 0 else self.R @ dirs
                    stack.append((seq + [dir], next_dirs))
            else:
                for d1 in [0, 1]:
                    for d2 in [0, 1]:
                        if d1 == 0 and d2 == 0:
                            next_dirs = self.LL @ dirs
                        elif d1 == 0 and d2 == 1:
                            next_dirs = self.RL @ dirs
                        elif d1 == 1 and d2 == 0:
                            next_dirs = self.LR @ dirs
                        else:  # d1 == 1 and d2 == 1
                            next_dirs = self.RR @ dirs

                        stack.append((seq + [d1, d2], next_dirs))

    def build_4bit_matrix_map(self):
        from itertools import product

        mat_map = {}
        for bits in product([0, 1], repeat=4):
            mat = self.L if bits[0] == 0 else self.R
            for b in bits[1:]:
                mat = (self.L if b == 0 else self.R) @ mat
            mat_map[bits] = mat
        return mat_map

    def simulate_non_recursive_4bit(self, A, B, start_time):
        matrix4_map = self.build_4bit_matrix_map()
        stack = [([], B)]

        while stack:
            now_time = time.time()
            if now_time - start_time > 60:
                raise TimeoutError("Simulation Timeout: 60s.")

            seq, dirs = stack.pop()

            if len(seq) == self.NUM_QUBITS:
                result = A @ dirs
                val = result.toarray()[0, 0]
                if abs(val) > 1e-10:
                    self.sim_result[tuple(seq)] = val
                continue

            remaining = self.NUM_QUBITS - len(seq)
            if remaining >= 4:
                for bits4, mat in matrix4_map.items():
                    next_seq = seq + list(bits4)
                    next_dirs = mat @ dirs
                    stack.append((next_seq, next_dirs))
            elif remaining >= 2:
                for d1 in [0, 1]:
                    for d2 in [0, 1]:
                        next_dirs = (
                            self.LL @ dirs if d1 == 0 and d2 == 0 else
                            self.RL @ dirs if d1 == 0 and d2 == 1 else
                            self.LR @ dirs if d1 == 1 and d2 == 0 else
                            self.RR @ dirs
                        )
                        stack.append((seq + [d1, d2], next_dirs))
            else:
                for d in [0, 1]:
                    next_dirs = self.L @ dirs if d == 0 else self.R @ dirs
                    stack.append((seq + [d], next_dirs))

            if self.DEBUG:
                print(time.time() - now_time, "seconds elapsed for current iteration:", seq)


    def simulation(self):
        if self.TYPE == 'numpy':
            import numpy as cp
            from scipy import sparse
        elif self.TYPE == 'cupy':
            import cupy as cp
            from cupyx.scipy import sparse

        A = cp.zeros((1, self.L.shape[0]))
        A[0, self.leaf] = 1
        A = sparse.csr_matrix(A, dtype=cp.complex128)

        B = cp.zeros((self.L.shape[0], 1))
        B[0, 0] = 1
        B = sparse.csr_matrix(B, dtype=cp.complex128)

        start_simulation = time.time()

        # self.recursion([], B, A, start_simulation)
        self.simulate_non_recursive_4bit(A, B, start_simulation)


    def __iter__(self):
        ...
        # A = cp.zeros((1, self.L.shape[0]))
        # A[0, self.leaf] = 1
        # A = sparse.csc_matrix(A, dtype=cp.complex128)

        # B = cp.zeros((self.L.shape[0], 1))
        # B[0, 0] = 1
        # B = sparse.csc_matrix(B, dtype=cp.complex128)

        # start_simulation = time.time()

        # self.recursion([], B, A, start_simulation)

        # for i in range(2**self.NUM_QUBITS):
        #     if time.time() - start_simulation > 60:
        #         raise TimeoutError("Simulation Timeout: 60s.")

        #     seq = [(i >> j) & 1 for j in reversed(range(self.NUM_QUBITS))]
        #     dirs = B
        #     if self.NUM_QUBITS <= 10:
        #         for dir in seq:
        #             dirs = self.L @ dirs if dir == 0 else self.R @ dirs
        #     else:
        #         for d1, d2 in zip(seq[::2], seq[1::2]):
        #             if d1 == 0 and d2 == 0:
        #                 dirs = self.LL @ dirs
        #             elif d1 == 0 and d2 == 1:
        #                 dirs = self.RL @ dirs
        #             elif d1 == 1 and d2 == 0:
        #                 dirs = self.LR @ dirs
        #             elif d1 == 1 and d2 == 1:
        #                 dirs = self.RR @ dirs

        #         if self.NUM_QUBITS % 2 == 1:
        #             d = seq[-1]
        #             dirs = self.L @ dirs if d == 0 else self.R @ dirs

        #     result = A @ dirs
            # val = result.toarray()[0, 0]
            # val = result.item()
            # print("L: ", self.L)
            # print("R: ", self.R)
            # print("Leaf: ", self.leaf)
            # print(f"Path: {seq}, Value: {val}")
            # if abs(val) > 1e-10:
            #     self.sim_result[tuple(seq)] = val
            # yield val, seq

    def results(self):
        return self.sim_result

def simulation_all_path(L, R, leaf, NUM_QUBITS, TYPE, DEBUG=True):
    sim = SimulationAllPath(L, R, leaf, NUM_QUBITS, TYPE, DEBUG=DEBUG)
    sim.simulation()
    # for val, seq in sim:
    #     pass  # Iterate through all paths to compute results
    return sim.results()

def show_matrix(matrix: List[List[complex]]):
    """
    Display a matrix in a readable format.
    """
    for row in matrix:
        print(" | ".join(f"{val:8.4f}" for val in row))
    print("\n")

class WeakSimulation:

    root_index = 0

    def __init__(self, 
                 L, 
                 R, 
                 Mapping: List[List[int]], 
                 NUM_QUBITS: int, 
                 TYPE: str = 'numpy', 
                 num_shots: int = 1024,
                 tol: float = 1e-10,
                 DEBUG: bool = True
    ):
        self.L = L
        self.R = R
        self.Mapping = Mapping
        self.NUM_QUBITS = NUM_QUBITS
        self.TYPE = TYPE
        self.num_shots = num_shots
        self.tol = tol
        self.DEBUG = DEBUG

        if self.TYPE == 'numpy':
            import numpy as np
            import scipy.sparse as sparse
            self.L = self.L.toarray()
            self.R = self.R.toarray()
        elif self.TYPE == 'cupy':
            import cupy as cp
            import cupyx.scipy.sparse as sparse
            self.L = self.L.toarray()
            self.R = self.R.toarray()
        else:
            raise ValueError("Unsupported TYPE. Use 'numpy' or 'cupy'.")


        self.one_terminals = Mapping[-2]
        self.zero_terminals = Mapping[-1]
        self.num_of_nodes = L.shape[0]
        self.root_index = 0

        self.adj_list = {}
        self.parent_map = {}
        self.build_adj_list_and_parent_map()

        self.sim_result = {}
        self.probs = {}
        self.downstream = {}
        self.upstream = {}

    def build_adj_list_and_parent_map(self):

        for i in range(self.num_of_nodes):
            self.adj_list[i] = {}
            self.parent_map[i] = []

        for i in range(self.num_of_nodes):
            for j in range(self.num_of_nodes):

                if abs(self.L[j, i]) > self.tol or abs(self.R[j, i]) > self.tol:
                    self.adj_list[i][j] = (float(abs(self.L[j, i]) ** 2), float(abs(self.R[j, i]) ** 2))
                    self.parent_map[j].append(i)


    def prune_edges(self, node_index: int = root_index) -> bool:

        if node_index in self.zero_terminals:
            return True
        if node_index in self.one_terminals:
            return False

        all_child_should_be_pruned = True

        child_nodes = list(self.adj_list[node_index].keys())

        for child_node in child_nodes:
            if self.prune_edges(child_node):
                del self.adj_list[node_index][child_node]
            else:
                all_child_should_be_pruned = False

        return all_child_should_be_pruned


    def compute_downstream_contributions(self, node_index: int = root_index) -> float:

        if node_index in self.one_terminals:
            return 1.0

        contributions = 0

        for child_node, (l_val, r_val) in self.adj_list[node_index].items():
            contributions += (l_val + r_val) * self.compute_downstream_contributions(child_node)

        self.downstream[node_index] = contributions

        return contributions


    def compute_upstream_contributions(self):

        from collections import deque

        Q = deque()
        Q.append(self.root_index)
        self.upstream[self.root_index] = 1.0

        # if self.DEBUG:
            # print("[WEAK_SIM] Initial Queue:", list(Q))
            # print("[WEAK_SIM] Initial Upstream Contributions:", self.upstream)
            # print("[WEAK_SIM] BFS Traversal Starts")

        while Q:
            node_index = Q.popleft()

            # if self.DEBUG: print(f"[WEAK_SIM] Processing Node: {node_index}")

            for child_node in self.adj_list[node_index]:
                Q.append(child_node)
            
            if node_index == self.root_index:
                continue

            contributions = 0

            for parent_node in self.parent_map[node_index]:
                l_val, r_val = self.adj_list[parent_node][node_index]
                contributions += (l_val + r_val) * self.upstream[parent_node]

            self.upstream[node_index] = contributions
        
        # if self.DEBUG:
        #     print("[WEAK_SIM] BFS Traversal Ends")
        #     print("[WEAK_SIM] Final Upstream Contributions:", self.upstream)

    def simulation_once(self) -> str:

        result = ""

        processing_nodes = []
        processing_nodes.append(self.root_index)

        for level in range(self.NUM_QUBITS):

            zero_prob = 0.0

            for node in processing_nodes:
                for child in self.adj_list[node]:
                    l_val = self.adj_list[node][child][0]
                    if l_val > 0:
                        zero_prob += l_val * self.upstream[node] * self.downstream[child]

            next_processing_nodes = []

            if random.random() < zero_prob:
                result += "0"
                for node in processing_nodes:
                    for child in self.adj_list[node]:
                        l_val = self.adj_list[node][child][0]
                        if l_val > self.tol:
                            next_processing_nodes.append(child)
            else:
                result += "1"
                for node in processing_nodes:
                    for child in self.adj_list[node]:
                        r_val = self.adj_list[node][child][1]
                        if r_val > self.tol:
                            next_processing_nodes.append(child)

            processing_nodes = next_processing_nodes

        return result

    def simulation(self):

        if self.DEBUG:
            print("[WEAK_SIM] L:")
            show_matrix(self.L)
            print("[WEAK_SIM] R:")
            show_matrix(self.R)
            print("[WEAK_SIM] adj_list:", self.adj_list)
            print("[WEAK_SIM] parent_map:", self.parent_map)
            print("[WEAK_SIM] Mapping:", self.Mapping)

        self.prune_edges()
        if self.DEBUG: print("[WEAK_SIM] adj_list after pruned:", self.adj_list)
        
        self.compute_downstream_contributions()
        if self.DEBUG: print("[WEAK_SIM] Downstream Contributions:", self.downstream)
        
        self.compute_upstream_contributions()
        if self.DEBUG: print("[WEAK_SIM] Upstream Contributions:", self.upstream)

        for _ in range(self.num_shots):

            result = self.simulation_once()

            if result in self.sim_result:
                self.sim_result[result] += 1
            else:
                self.sim_result[result] = 1

    def get_result(self) -> Dict[str, int]:

        if not self.sim_result:
            self.simulation()
        return self.sim_result

    def compute_probability(self):

        for key, value in self.sim_result.items():
            self.probs[key] = value / self.num_shots
        return self.probs

    def get_probs(self) -> Dict[str, float]:

        if not self.probs:
            if not self.sim_result:
                self.simulation()
            self.compute_probability()
        return self.probs

def weak_simulation(L, R, Mapping: List[List[int]], NUM_QUBITS: int, TYPE: str = 'numpy', num_shots: int = 1024, tol: float = 1e-10):

    weak_sim = WeakSimulation(L, R, Mapping, NUM_QUBITS, TYPE, num_shots, tol)
    return weak_sim.get_result()

def dd_sparse(operations: List[GateOperation], 
              MATRIX_SIZE: int, 
              NUM_QUBITS: int, 
              TYPE: str, 
              MODE: str = 'strong_simulation', 
              DEBUG: bool = True
) -> Tuple[List[List[complex]], List[List[complex]], List[List[int]]]:

    L, R, Mapping = initialization_sparse(MATRIX_SIZE, NUM_QUBITS, TYPE)

    total_start = time.time()

    if TYPE == 'numpy':
        import numpy as cp
        from scipy import sparse
    elif TYPE == 'cupy':
        import cupy as cp
        from cupyx.scipy import sparse

    def apply_the_unitary(L, R, op, Mapping):

        MATRIX_SIZE = L.shape[0]

        A = cp.ones(MATRIX_SIZE, dtype=complex)
        A[Mapping[op.target]] = op.a
        B = cp.zeros(MATRIX_SIZE, dtype=complex)
        B[Mapping[op.target]] = op.b
        C = cp.zeros(MATRIX_SIZE, dtype=complex)
        C[Mapping[op.target]] = op.c
        D = cp.ones(MATRIX_SIZE, dtype=complex)
        D[Mapping[op.target]] = op.d

        A = sparse.diags(A, offsets=0, format='csr')
        B = sparse.diags(B, offsets=0, format='csr')
        C = sparse.diags(C, offsets=0, format='csr')
        D = sparse.diags(D, offsets=0, format='csr')

        L_U = L @ A + R @ B
        R_U = L @ C + R @ D

        return L_U, R_U

    def make_keep_mask(control, Mapping, MATRIX_SIZE_OLD):

        keep = cp.zeros(MATRIX_SIZE_OLD, dtype=int)
        keep_complement = cp.ones(MATRIX_SIZE_OLD, dtype=int)
        keep[Mapping[control]] = 1
        keep_complement[Mapping[control]] = 0
        D = sparse.diags(keep, offsets=0, format='csr')
        D_complement = sparse.diags(keep_complement, offsets=0, format='csr')
        return D, D_complement

    def expand_unitaries(control, target, MATRIX_SIZE_OLD, L, R, L_U, R_U, D, D_complement):

        if control < target:

            R_masked = R @ D_complement
            R_U_masked = R_U[:, :MATRIX_SIZE_OLD] @ D

            L_U = sparse.bmat([[L, None], 
                               [None, L_U]], format='csr')

            R_U = sparse.bmat([[R_masked, None],
                               [R_U_masked, R_U]], format='csr')

        else:
            L_U_masked = L_U @ D_complement
            L_masked = L[:, :MATRIX_SIZE_OLD] @ D
            L_U = sparse.bmat([[L_U_masked, None],
                               [L_masked, L]], format='csr')

            R_U = sparse.bmat([[R_U, None],
                               [None, R]], format='csr')

        return L_U, R_U

    def compute_reachable(L, R, MATRIX_SIZE):

        L_B = L.astype(bool).astype(float)
        R_B = R.astype(bool).astype(float)

        LR = L_B + R_B
        LR = LR.astype(bool)
        LR = LR.astype(float)

        traverse = sparse.csr_matrix((MATRIX_SIZE, 1), dtype=bool)
        traverse[0, 0] = True
        prev = -1
        while traverse.nnz != prev:
            prev = traverse.nnz
            traverse = traverse.astype(float)
            traverse = (traverse + LR @ traverse)
            traverse = traverse.astype(bool)

        traverse = traverse.astype(float)
        traverse = set(traverse.toarray().ravel().nonzero()[0].tolist())

        return traverse

    def update_mapping(Mapping, traverse, MATRIX_SIZE, MATRIX_SIZE_OLD):

        Mapping_U = [
            sorted(
                [n + MATRIX_SIZE_OLD * i
                for i in range(1, MATRIX_SIZE // MATRIX_SIZE_OLD + 1)
                for n in mapping
                if n + MATRIX_SIZE_OLD * i in traverse] +
                [n for n in mapping if n in traverse]
            )
            for mapping in Mapping
        ]

        mapping_middle = time.time()

        traverse_list = list(traverse)
        traverse_dict = {i: idx for idx, i in enumerate(traverse_list)}

        Mapping = [[traverse_dict[node] for node in mapping_U] for mapping_U in Mapping_U]

        return Mapping, mapping_middle

    def project_matrices(L, R, traverse, MATRIX_SIZE):

        not_traverse = cp.array(list(set(range(MATRIX_SIZE)) - traverse))

        rows = cp.arange(MATRIX_SIZE)
        rows = rows[~cp.isin(rows, not_traverse)]
        one = cp.ones_like(rows, dtype=cp.int32)

        S = sparse.csr_matrix((one, (cp.arange(len(rows)), rows)),
                                shape=(len(rows), MATRIX_SIZE), dtype=cp.complex128)

        L = S @ L @ S.T
        R = S @ R @ S.T

        return L, R

    def handle_controlled_gate(L, R, L_U, R_U, op, Mapping):

        MATRIX_SIZE = L.shape[0]
        MATRIX_SIZE_OLD = MATRIX_SIZE

        for control in reversed(op.control):

            D, D_complement = make_keep_mask(control, Mapping, MATRIX_SIZE_OLD)
            L_U, R_U = expand_unitaries(control, op.target, MATRIX_SIZE_OLD, L, R, L_U, R_U, D, D_complement) 
            MATRIX_SIZE += MATRIX_SIZE_OLD

        L, R = L_U, R_U

        if DEBUG:
            print("L: ", L)
            print("R: ", R)

        traverse_start = time.time()
        traverse = compute_reachable(L, R, MATRIX_SIZE)

        mapping_start = time.time()
        Mapping, mapping_middle = update_mapping(Mapping, traverse, MATRIX_SIZE, MATRIX_SIZE_OLD)

        construct_start = time.time()
        L, R = project_matrices(L, R, traverse, MATRIX_SIZE)

        if not L.shape == R.shape:
            raise ValueError("Size of L and R not equal")

        return L, R, Mapping

    def simulation(MODE):

        if MODE == 'strong_simulation':
            sim_result = simulation_all_path(L, R, Mapping[-2], NUM_QUBITS, TYPE, DEBUG=DEBUG)
        elif MODE == 'weak_simulation':
            sim_result = weak_simulation(L, R, Mapping, NUM_QUBITS, TYPE, num_shots=1024)
        elif MODE is None:
            raise ValueError("MODE must be specified. Use 'strong_simulation' or 'weak_simulation'.")
        else:
            raise ValueError("Unsupported MODE. Use 'strong_simulation' or 'weak_simulation'.")
        return sim_result

    for op in operations:

        if time.time() - total_start > 60:
            raise TimeoutError("Timeout: 60s.")

        L_U, R_U = apply_the_unitary(L, R, op, Mapping)

        if isinstance(op, SingleQubitGate):  
            L, R = L_U, R_U
        elif isinstance(op, ControlledGate):     
            L, R, Mapping = handle_controlled_gate(L, R, L_U, R_U, op, Mapping)

        if DEBUG:
            print("MATRIX_SIZE: ", L.shape[0])
            print("L: ", L)
            print("R: ", R)
            print("Mapping: ", Mapping)

    sim_result = simulation(MODE)

    return sim_result

def dd_sparse_numpy(operations: List[GateOperation], INIT_MATRIX_SIZE: int, NUM_QUBITS: int, MODE: str = 'strong_simulation', DEBUG: bool = True):

    def monitor_cpu_usage(stop_event, result_dict, interval=0.5):
        pid = os.getpid()
        process = psutil.Process(pid)
        cpu_usages = []
        mem_usages = []

        while not stop_event.is_set():
            cpu = process.cpu_percent(interval=None)  # percent of a core
            mem = process.memory_info().rss / 1024**2  # in MB
            cpu_usages.append(cpu)
            mem_usages.append(mem)
            time.sleep(interval)

        result_dict["avg_cpu"] = sum(cpu_usages)/len(cpu_usages) if cpu_usages else 0.0
        result_dict["peak_mem"] = max(mem_usages) if mem_usages else 0.0

    import numpy as cp
    import scipy.sparse as sparse
    try:
        result_dict = {}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_cpu_usage, args=(stop_event, result_dict))

        np_start = time.time()
        monitor_thread.start()
        # for i in range(1024):
            # if time.time() - np_start > 60:
            #     raise TimeoutError("Simulation Timeout: 60s.")
        sim_result = dd_sparse(operations, INIT_MATRIX_SIZE, NUM_QUBITS, TYPE='numpy', MODE=MODE, DEBUG=DEBUG)

        np_end = time.time()

        stop_event.set()
        monitor_thread.join()

        result_dict["time_spent"] = np_end - np_start
        result_dict["error"] = None

        if DEBUG: print(sim_result)
        # for comparison function)
        result_dict["sim_result"] = sim_result
        # print("|0>*n: ", compute_result(L, R, [0]*NUM_QUBITS, Mapping[-2]))
        # print("|1>*n: ", compute_result(L, R, [1]*NUM_QUBITS, Mapping[-2]))
    except Exception as e:
        np_end = time.time()
        stop_event.set()
        monitor_thread.join()
        # print("Error in numpy + scipy:", e)
        result_dict["time_spent"] = np_end - np_start
        result_dict["error"] = str(e)
        traceback.print_exc()

    # print("Time spent: ", np_end - np_start)
    return result_dict


