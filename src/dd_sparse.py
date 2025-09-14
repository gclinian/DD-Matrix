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
    def __init__(self, L, R, leaf, NUM_QUBITS, TYPE, DEBUG=False):
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

def simulation_all_path(L, R, leaf, NUM_QUBITS, TYPE, DEBUG=False):
    sim = SimulationAllPath(L, R, leaf, NUM_QUBITS, TYPE, DEBUG=DEBUG)
    sim.simulation()
    # for val, seq in sim:
    #     pass  # Iterate through all paths to compute results
    return sim.results()

def show_adj_list(adj_list: Dict[int, Dict[int, Tuple[float, float]]]):
    """
    Display the adjacency list in a readable format.
    
    Args:
        adj_list (Dict[int, Dict[int, Tuple[float, float]]]): 
            The adjacency list to display.
    """
    for node, edges in adj_list.items():
        print(f"Node {node}:")
        for child, (l_val, r_val) in edges.items():
            print(f"-> Child {child}: L={l_val:.2f}, R={r_val:.2f}")

def show_matrix(matrix: List[List[complex]]):
    """
    Display a matrix in a readable format.
    
    Args:
        matrix (List[List[complex]]): 
            The matrix to display.
    """
    for row in matrix:
        print(" | ".join(f"{val:.2f}" for val in row))
    print("-" * 30)

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
            import numpy as cp
            import scipy.sparse as sparse
            
        elif self.TYPE == 'cupy':
            import cupy as cp
            import cupyx.scipy.sparse as sparse

        else:
            raise ValueError("Unsupported TYPE. Use 'numpy' or 'cupy'.")

        self.cp = cp
        self.sparse = sparse

        self.one_terminals = Mapping[-2]
        self.zero_terminals = Mapping[-1]
        self.num_of_nodes = L.shape[0]
        self.root_index = 0

        self.adj_list = {}
        self.parent_map = {}

        self.sim_result = {}
        self.probs = {}
        self.downstream = {}
        self.upstream = {}

        self.reachable_set = set()
        self.pruned_set = set()
        assert(self.NUM_QUBITS + 2 == len(self.Mapping))
        self.my_Mapping = [[] for _ in range(self.NUM_QUBITS + 2)]

    def build_adj_list(self):
        
        if self.DEBUG:
            print("[WEAK_SIM] Building adjacency list and parent map...")
            print("[WEAK_SIM] Number of nodes:", self.num_of_nodes)

        from collections import defaultdict

        L_coo = self.L.tocoo()
        R_coo = self.R.tocoo()

        edges = defaultdict(lambda: [0.0, 0.0])

        for j, i, v in zip(L_coo.row, L_coo.col, L_coo.data):    
            edges[(int(i), int(j))][0] = float(abs(v) ** 2)

        for j, i, v in zip(R_coo.row, R_coo.col, R_coo.data):    
            edges[(int(i), int(j))][1] = float(abs(v) ** 2)


        if self.DEBUG:
            print("[WEAK_SIM] Edges found:", len(edges))

        for (i, j), (wL, wR) in edges.items():
            
            if i in self.adj_list:
                self.adj_list[i][j] = (wL, wR)
            else:
                self.adj_list[i] = {j: (wL, wR)}
            
            if j not in self.adj_list:
                self.adj_list[j] = {}

    def check_direction(self, Mapping):
        level = {}
        for lvl, nodes in enumerate(Mapping):
            for n in nodes:
                if n in level and level[n] != lvl:
                    print(f"[Error] Node {n} appears in multiple levels: {level[n]} and {lvl}.")
                level[n] = lvl

        bad_edges = []
        
        miss_node_in_mapping = False

        for p, children in self.adj_list.items():
            if p not in level:
                print(f"[Error] Parent node {p} not found in any level of the mapping.")
                miss_node_in_mapping = True
                continue
            for c in children:
                if c not in level:
                    print(f"[Error] Child node {c} not found in any level of the mapping.")
                    miss_node_in_mapping = True
                    continue
                if level[p] >= level[c] and c not in Mapping[-2] and c not in Mapping[-1]:
                    bad_edges.append((p, c, level[p], level[c]))
        
        no_error = True
        if miss_node_in_mapping:
            print("[Error] Some nodes are missing in the mapping levels.")
            no_error = False
        
        if len(bad_edges) > 0:
            print("[Error] Bad edges found in the adjacency list:")
            for p, c, lvl_p, lvl_c in bad_edges:
                print(f"Parent {p} (level {lvl_p}) -> Child {c} (level {lvl_c})")
            print("[Error] The adjacency list does not respect the mapping levels.")
            no_error = False
        
        if no_error:
            print("[INFO] All edges in the adjacency list respect the mapping levels.")
    
    def prune_edges(self, node_index: int = root_index) -> bool:

        if self.DEBUG:
            print(f"[WEAK_SIM] Pruning edges for node {node_index}")
        
        if node_index in self.zero_terminals:
            self.pruned_set.add(node_index)
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

        if all_child_should_be_pruned:
            self.pruned_set.add(node_index)
        
        return all_child_should_be_pruned

    def build_parent_map(self):
        """
        Build a parent map for the adjacency list.
        This map will help in computing upstream contributions.
        """

        for parent_node, children in self.adj_list.items():
            for child_node in children:
                if child_node not in self.parent_map:
                    self.parent_map[child_node] = []
                self.parent_map[child_node].append(parent_node)
    
    def build_reachable_set(self, node_index: int = root_index):
        """
        Build a set of reachable nodes starting from the root node.
        This is used to ensure that we only compute contributions for reachable nodes.
        """

        if node_index in self.reachable_set:
            return

        self.reachable_set.add(node_index)

        for child_node in self.adj_list[node_index]:
            self.build_reachable_set(child_node)

    def check_loop(self):
        """
        Check if there are any loops in the adjacency list.
        This is used to ensure that the graph is acyclic.
        """

        visited = set()
        stack = set()

        def visit(node_index):
            if node_index in stack:
                print(f"[Error] Loop detected at node {node_index}")
                return True
            
            if node_index in visited:
                return False
            
            visited.add(node_index)
            stack.add(node_index)

            for child_node in self.adj_list.get(node_index, {}):
                if visit(child_node):
                    return True
            
            stack.remove(node_index)
            return False

        for node_index in self.adj_list:
            if visit(node_index):
                return True
        
        print("[INFO] No loops detected.")
        return False

    def build_my_mapping(self, node_index: int = root_index, level: int = 0):
        
        if node_index not in self.my_Mapping[level]:
            if node_index in self.Mapping[-1]:
                self.my_Mapping[-1].append(node_index)
            elif node_index in self.Mapping[-2]:
                self.my_Mapping[-2].append(node_index)
            else:
                self.my_Mapping[level].append(node_index)
        else:
            return


        if node_index in self.adj_list:
            for child_node in self.adj_list[node_index]:
                self.build_my_mapping(child_node, level + 1)
    
    def check_two_mappings_are_equal(self):
        """
        Check if the two mappings are equal.
        This is used to ensure that the mapping is consistent with the adjacency list.
        """
        are_equal = True

        for i in range(len(self.my_Mapping)):
            if set(self.my_Mapping[i]) != set(self.Mapping[i]):
                # print("[Error] Mappings are not equal at level", i)
                are_equal = False
        
        if are_equal:
            print("[INFO] Mappings are equal.")
        else:
            print("[Error] Mappings are not equal.")
    
    def check_mapping_and_parent_map(self, Mapping: List[List[int]]) -> bool:
        """
        Check if the mapping and parent map are consistent.
        This is used to ensure that the parent map is correctly built from the adjacency list.
        """

        consistent = True
        
        for level in range(1, len(Mapping) - 1):
            for node_index in Mapping[level]:
                if node_index not in self.parent_map:
                    print(f"[Error] Node {node_index} at level {level} is not in parent_map.")
                    consistent = False
                else:
                    for parent_node in self.parent_map[node_index]:
                        if parent_node not in Mapping[level - 1]:
                            print(f"[Error] Parent node {parent_node} of {node_index} is not in the previous level mapping.")
                            consistent = False
        
        for node_index in Mapping[-1]:
            if node_index not in self.parent_map:
                    print(f"[Error] Node {node_index} at level {level} is not in parent_map.")
                    consistent = False
            else:
                for parent_node in self.parent_map[node_index]:
                    if parent_node not in Mapping[-3]:
                        print(f"[Error] Parent node {parent_node} of {node_index} is not in the previous level mapping.")
                        consistent = False
        
        return consistent
    
    def filter_adj_list(self):
        """
        Filter the adjacency list to only include reachable nodes.
        This is used to optimize the simulation by ignoring unreachable nodes.
        """

        filtered_adj_list = {}
        for node_index in self.reachable_set:
            if node_index in self.adj_list:
                filtered_adj_list[node_index] = self.adj_list[node_index]
        
        self.adj_list = filtered_adj_list

    def filter_parent_map(self):
        """
        Filter the parent map to only include reachable nodes.
        This is used to optimize the upstream contributions computation.
        """

        filtered_parent_map = {}
        for node_index in self.reachable_set:
            if node_index in self.parent_map:
                filtered_parent_map[node_index] = list(set(self.parent_map[node_index]) & self.reachable_set)
        
        self.parent_map = filtered_parent_map
    
    def check_node(self, node_index: int):
        """
        Check if a node is reachable and exists in the adjacency list.
        If not, raise an error.
        """

        print(f"\n[Error] Checking node:, {node_index}")

        if node_index not in self.reachable_set:
            print(f"[Error] Node {node_index} is not reachable from the root node.")
        else:
            print(f"[Error] Node {node_index} is reachable from the root node.")
            
        if node_index in self.pruned_set:
            print(f"[Error] Node {node_index} has been pruned.")
        else:
            print(f"[Error] Node {node_index} hasn't been pruned.")

        if node_index not in self.adj_list:
            print(f"[Error] Node {node_index} does not exist in the adjacency list.")
        else:
            print(f"[Error] Children of node {node_index}: {list(self.adj_list[node_index].keys())}")
        
        if node_index not in self.parent_map:
            print(f"[Error] Node {node_index} is not in parent_map.")
        else:
            print(f"[Error] Node {node_index} is in parent_map.")
        
    def compute_downstream_contributions(self, node_index: int = root_index) -> float:

        if node_index in self.one_terminals:
            self.downstream[node_index] = 1.0
            return 1.0

        if node_index in self.zero_terminals:
            self.downstream[node_index] = 0.0
            return 0.0

        contributions = 0

        for child_node, (l_val, r_val) in self.adj_list[node_index].items():
            contributions += (l_val + r_val) * self.compute_downstream_contributions(child_node)

        self.downstream[node_index] = contributions

        return contributions

    def compute_downstream_contributions_with_stack(self):
        
        from collections import deque

        S = deque()
        S.append(self.root_index)

        while S:
            node_index = S.pop()

            if node_index in self.one_terminals:
                self.downstream[node_index] = 1.0
                continue

            if node_index in self.zero_terminals:
                self.downstream[node_index] = 0.0
                continue

            if node_index in self.downstream:
                continue

            all_children_computed = True
            contributions = 0.0

            for child_node, (l_val, r_val) in self.adj_list[node_index].items():
                if child_node not in self.downstream:
                    all_children_computed = False
                    S.append(child_node)
                else:
                    contributions += (l_val + r_val) * self.downstream[child_node]
            
            if all_children_computed:
                self.downstream[node_index] = contributions
            else:
                S.append(node_index)

    def compute_upstream_contributions(self):

        from collections import deque

        Q = deque()
        Q.append(self.root_index)
        self.upstream[self.root_index] = 1.0

        miss_compute = False

        while Q:
            node_index = Q.popleft()

            if node_index not in self.one_terminals and node_index not in self.zero_terminals:
                for child_node in self.adj_list[node_index]:
                    Q.append(child_node)
            
            if node_index in self.upstream:
                continue

            contributions = 0

            for parent_node in self.parent_map[node_index]:
                if parent_node not in self.upstream:
                    # self.check_node(parent_node)
                    if not miss_compute:
                        print(f"[Error] Upstream of node {parent_node}, ", end="")
                        miss_compute = True
                    else:
                        print(f"{parent_node}, ", end="")
                    continue
                l_val, r_val = self.adj_list[parent_node][node_index]
                contributions += (l_val + r_val) * self.upstream[parent_node]

            self.upstream[node_index] = contributions
        
        if miss_compute:
            print("have not been computed")

    def compute_upstream_contributions_with_mapping(self, Mapping: List[List[int]]):
        """
        Compute upstream contributions using the Mapping.
        This is used to ensure that we correctly compute contributions for each node.
        """

        miss_compute = False

        for level in range(len(Mapping)):
            for node in Mapping[level]:
                if node == self.root_index:
                    self.upstream[node] = 1.0
                    continue
                
                self.upstream[node] = 0.0

                for parent_node in self.parent_map[node]:
                    if parent_node not in self.upstream:
                        if not miss_compute:
                            print(f"[Error] Upstream of node {parent_node}, ", end="")
                            miss_compute = True
                        else:
                            print(f"{parent_node}, ", end="")
                        continue
                    
                    l_val, r_val = self.adj_list[parent_node][node]
                    self.upstream[node] += (l_val + r_val) * self.upstream[parent_node]
        
        if miss_compute:
            print("have not been computed")

    def simulation_once(self) -> str:

        result = ""

        processing_nodes = []
        processing_nodes.append(self.root_index)

        for level in range(self.NUM_QUBITS):

            zero_prob = 0.0

            for node in processing_nodes:
                for child in self.adj_list[node]:
                    
                    if node not in self.reachable_set or child not in self.reachable_set:
                        raise ValueError(f"Node {node} or child {child} is not reachable from the root node.")
                    
                    l_val = self.adj_list[node][child][0]
                    if l_val > 0:
                        zero_prob += l_val * self.upstream.get(node, 0) * self.downstream.get(child, 0)

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

    def simulation_once_matrix(self) -> str:
        """
        Matrix-based weak simulation following the given pseudocode:
          U <- e_root
          L2 = |L|^2, R2 = |R|^2
          d  = downstream as a vector (diagonal of D)
          For each qubit:
            U0 = L2 @ U
            U1 = R2 @ U
            prob0 = d^T U0
            prob1 = d^T U1
            sample x in {0,1} with probs {prob0/prev_prob, prob1/prev_prob}
            if x == 0: U <- U0; prev_prob <- prob0
            else:      U <- U1; prev_prob <- prob1
        """
        cp = self.cp
        sparse = self.sparse
        import random

        # --- Lazy precompute |L|^2 and |R|^2 (cached) ---
        if not hasattr(self, "_L2") or self._L2 is None:
            L2 = self.L.copy().tocsr()
            R2 = self.R.copy().tocsr()
            # elementwise |.|^2 on data
            L2.data = cp.abs(L2.data) ** 2
            R2.data = cp.abs(R2.data) ** 2
            # ensure real dtype for safety
            if getattr(L2.data, "dtype", None) is not None and L2.data.dtype.kind != 'f':
                L2.data = L2.data.astype(cp.float64)
                L2 = L2.astype(cp.float64)
            if getattr(R2.data, "dtype", None) is not None and R2.data.dtype.kind != 'f':
                R2.data = R2.data.astype(cp.float64)
                R2 = R2.astype(cp.float64)
            self._L2 = L2
            self._R2 = R2
        else:
            L2 = self._L2
            R2 = self._R2

        # --- Lazy precompute downstream vector d (cached) ---
        if not hasattr(self, "_dvec") or self._dvec is None:
            # make sure downstream is available
            if not self.downstream:
                self.compute_downstream_contributions()
            dvec = cp.zeros(self.num_of_nodes, dtype=cp.float64)
            keys = list(self.downstream.keys())
            vals = [self.downstream[k] for k in keys]
            if len(keys) > 0:
                dvec[cp.asarray(keys, dtype=cp.int64)] = cp.asarray(vals, dtype=cp.float64)
            self._dvec = dvec
        else:
            dvec = self._dvec

        # --- Initialize U at root ---
        U = cp.zeros(self.num_of_nodes, dtype=cp.float64)
        U[self.root_index] = 1.0

        # robust init (equals 1 if graph is normalized)
        prev_prob = float(cp.dot(self._dvec, U))

        bits = []
        for _ in range(self.NUM_QUBITS):
            U0 = L2.dot(U)
            U1 = R2.dot(U)

            prob0 = float(cp.dot(self._dvec, U0))
            prob1 = float(cp.dot(self._dvec, U1))

            denom = max(prev_prob, self.tol)
            p0 = prob0 / denom
            # numerical guards
            if p0 < 0.0:
                p0 = 0.0
            elif p0 > 1.0:
                p0 = 1.0

            if random.random() < p0:
                bits.append("0")
                U = U0
                prev_prob = prob0
            else:
                bits.append("1")
                U = U1
                prev_prob = prob1

        return "".join(bits)

    def simulation(self):

        self.build_adj_list()

        self.build_parent_map()

        if self.DEBUG:
            print("[WEAK_SIM] adj_list:")
            show_adj_list(self.adj_list)
            print("[WEAK_SIM] Mapping:")
            print(self.Mapping)
            print("[WEAK_SIM] Parent Map:", self.parent_map)

        self.check_direction(self.Mapping)

        

        # if self.DEBUG: print("[WEAK_SIM] Pruning Edges...")
        
        # self.prune_edges()
        
        # if self.DEBUG: print("[WEAK_SIM] Edges after pruning:")
        # if self.DEBUG: show_adj_list(self.adj_list)
        
        
        
        self.build_reachable_set()

        self.check_loop()
        
        if self.DEBUG: print("[WEAK_SIM] Reachable Set:", self.reachable_set)
        
        self.build_my_mapping()

        self.check_two_mappings_are_equal()
        
        print("[WEAK_SIM] Checking Mapping and Parent Map consistency...")
        if self.check_mapping_and_parent_map(self.Mapping): 
            print("[Info] Mapping and Parent Map are consistent.")
        else:
            print("[Error] Mapping and Parent Map are NOT consistent.")
            
        print("[WEAK_SIM] Checking My Mapping and Parent Map consistency...")
        if self.check_mapping_and_parent_map(self.my_Mapping):
            print("[Info] My Mapping and Parent Map are consistent.")
        else:
            print("[Error] My Mapping and Parent Map are NOT consistent.")


        # self.filter_adj_list()
        # if self.DEBUG: print("[WEAK_SIM] Filtered Adjacency List:")
        # if self.DEBUG: show_adj_list(self.adj_list)

        # self.filter_parent_map()
        # if self.DEBUG: print("[WEAK_SIM] Filtered Parent Map:", self.parent_map)
        
        if self.DEBUG: print("[WEAK_SIM] Computing Downstream Contributions...")
        
        self.compute_downstream_contributions_with_stack()
        
        if self.DEBUG: print("[WEAK_SIM] Downstream Contributions:", self.downstream)
        
        # if self.DEBUG: print("[WEAK_SIM] Computing Upstream Contributions...")
        
        # self.compute_upstream_contributions()

        # self.compute_upstream_contributions_with_mapping(self.my_Mapping)
        
        # if self.DEBUG: print("[WEAK_SIM] Upstream Contributions:", self.upstream)

        if self.DEBUG: print("[WEAK_SIM] Starting Simulation...")

        for _ in range(self.num_shots):

            result = self.simulation_once_matrix()

            if result in self.sim_result:
                self.sim_result[result] += 1
            else:
                self.sim_result[result] = 1
        
        if self.DEBUG: print("[WEAK_SIM] Simulation Results:", self.sim_result)

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

def check_matrix_and_mapping_consistency(L, R, Mapping: List[List[int]], NUM_QUBITS: int, TYPE: str = 'numpy'):
    """
    Check if the matrix and mapping are consistent.
    This is used to ensure that the adjacency list can be built correctly.
    """
    print("[Checking] L:")
    print(L)
    print("[Checking] R:")
    print(R)
    Checking = WeakSimulation(L, R, Mapping, NUM_QUBITS, TYPE)
    Checking.build_adj_list()
    print("\n[Checking] Adjacency List:")
    show_adj_list(Checking.adj_list)
    print("\n[Checking] Mapping:")
    print(Checking.Mapping)
    print("\n[Checking] Checking direction:")
    Checking.check_direction(Mapping)

def dd_sparse(operations: List[GateOperation], MATRIX_SIZE: int, NUM_QUBITS: int, TYPE: str, MODE: str = 'strong_simulation', DEBUG = False):
    L, R, Mapping = initialization_sparse(MATRIX_SIZE, NUM_QUBITS, TYPE)
    # print("Init Mapping: ", Mapping)
    # print("L:", L)
    # print("R:", R)
    total_start = time.time()

    if TYPE == 'numpy':
        import numpy as cp
        from scipy import sparse
    elif TYPE == 'cupy':
        import cupy as cp
        from cupyx.scipy import sparse
    
    for op in operations:
        # if time.time() - total_start > 60:
        #     raise TimeoutError("Timeout: 60s.")
        
        MATRIX_SIZE = L.shape[0]
        # print("MATRIX_SIZE: ", MATRIX_SIZE)
        # print("L: ", L)
        # print("R: ", R)
        L_U, R_U = copy.deepcopy(L), copy.deepcopy(R)
        
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

        if isinstance(op, SingleQubitGate):
            L, R = L_U, R_U
            # print("Mapping: ", Mapping)
        
        elif isinstance(op, ControlledGate):
            MATRIX_SIZE_OLD = MATRIX_SIZE
            for control in op.control:
                keep_U = cp.ones(MATRIX_SIZE, dtype=int)
                keep = cp.ones(MATRIX_SIZE_OLD, dtype=int)
                keep_U[Mapping[control]] = 0
                keep[Mapping[control]] = 0
                M_U = sparse.diags(keep_U, offsets=0, format='csr')
                M = sparse.diags(keep, offsets=0, format='csr')

                # print("M_U: ", M_U)
                # print("M: ", M)
                # print("L: ", L)
                # print("L[1:, 0]: ", L[1:, 0].shape)
                # print("L[1:, 1:]: ", L[1:, 1:].shape)
                # print("L_U @ M_U: ", (L_U @ M_U).shape)

                L_U_old = L_U @ M_U

                L_U = sparse.bmat([[L_U_old[:, 0:1], L_U_old[:, 1:], None, None],
                                    [L[:, 0:1], None, None, L[:, :]]], format='csr')

                R_old = R @ M
                # print("R_old: ", R_old)
                # print("R_old[1:, 0]: ", R_old[1:, 0:1])
                # print("R_old[1:, 1:]: ", R_old[1:, 1:])

                R_U = sparse.bmat([[R_U[:, 0:1], R_U[:, 1:], None, None],
                                    [R_old[:, 0:1], None, None, R_old[:, :]]], format='csr')

                # if control < op.target:
                #     R_masked = R @ M
                #     R_U_masked = R_U[:, :MATRIX_SIZE_OLD] @ M_U

                #     L_U = sparse.bmat([[L, None],
                #         [None, L_U]], format='csr')

                #     # print("Mapping: ", Mapping[control])

                #     # print("D: ", D)
                #     # print("D_complement: ", D_complement)
                #     # print("R: ", R)
                #     # print("R_U: ", R_U)
                #     # print("R_U[]: ", R_U[:, :MATRIX_SIZE_OLD])
                #     # print("R_masked: ", R_masked)
                #     # print("R_U_masked: ", R_U_masked)


                #     # if TYPE == 'cupy':
                #     #     print("R Masked Type: ", type(R_masked))
                #     #     print("R U Masked Type: ", type(R_U_masked))
                #     #     print("R U Type: ", type(R_U))
                #     #     print("R Masked shape: ", R_masked.shape)
                #     #     print("R U Masked shape: ", R_U_masked.shape)
                #     #     print("R_U shape: ", R_U.shape)
                #         # print("R_masked max row/col:", R_masked.nonzero())
                #         # print("R_U_masked max row/col:", R_U_masked.nonzero())
                #         # print("R_U max row/col:", R_U.nonzero())

                #     R_U = sparse.bmat([[R_masked, None],
                #         [R_U_masked, R_U]], format='csr')

                #     # print("L_U: ", L_U)
                #     # print("R_U: ", R_U)
                # else:

                #     L_U_masked = L_U @ D_complement
                #     L_masked = L[:, :MATRIX_SIZE_OLD] @ D
                #     L_U = sparse.bmat([[L_U_masked, None],
                #         [L_masked, L]], format='csr')

                #     R_U = sparse.bmat([[R_U, None],
                #         [None, R]], format='csr')

                MATRIX_SIZE += MATRIX_SIZE_OLD

            L, R = L_U, R_U
            
            # print("start traverse")
            if DEBUG:
                print("L: ", L)
                print("R: ", R)
            traverse_start = time.time()

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

            # print("Traverse: ", traverse)
            # print("end traverse, time spent:", time.time() - traverse_start)

            # print("start mapping")
            mapping_start = time.time()
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
            # print("Mapping_U: ", Mapping_U)
            
            # Check L, R, Mapping_U
            print("-"*100)
            print("[DD_SPARSE] Checking matrix and mapping consistency before mapping:")
            check_matrix_and_mapping_consistency(L, R, Mapping_U, NUM_QUBITS, TYPE)

            mapping_middle = time.time()
            # print("Mapping middle time spent:", mapping_middle - mapping_start)
            traverse_list = list(traverse)
            traverse_dict = {i: idx for idx, i in enumerate(traverse_list)}
            Mapping = [[traverse_dict[node] for node in mapping_U] for mapping_U in Mapping_U]
            
            # Checking L, R, Mapping
            # print("-"*100)
            # print("[DD_SPARSE] Checking matrix and mapping consistency after mapping:")
            # check_matrix_and_mapping_consistency(L, R, Mapping, NUM_QUBITS, TYPE)
            # print("Mapping_U after dict: ", Mapping_U)
            # Mapping = copy.deepcopy(Mapping_U)
            # print("end mapping, time spent:", time.time() - mapping_middle)  

            # print("start construct")
            construct_start = time.time()


            not_traverse = cp.array(list(set(range(MATRIX_SIZE)) - traverse))

            rows = cp.arange(MATRIX_SIZE)
            rows = rows[~cp.isin(rows, not_traverse)]
            one = cp.ones_like(rows, dtype=cp.int32)


            S = sparse.csr_matrix((one, (cp.arange(len(rows)), rows)),
                                    shape=(len(rows), MATRIX_SIZE), dtype=cp.complex128)
            
            L = S @ L @ S.T
            R = S @ R @ S.T

            # print("end construct, time spent:", time.time() - construct_start)

            MATRIX_SIZE = L.shape[0]

            print("-"*100)
            print("[DD_SPARSE] Checking matrix and mapping consistency after construction:")
            print("Gate: ", op.gate.name, op.target, "Control:", op.control)
            check_matrix_and_mapping_consistency(L, R, Mapping, NUM_QUBITS, TYPE)
            print("-"*100)
            

        if not L.shape == R.shape:
            raise ValueError("Size of L and R not equal")
        
        if DEBUG:
            print("MATRIX_SIZE: ", MATRIX_SIZE)
            print("L: ", L)
            print("R: ", R)
            print("Mapping: ", Mapping)
    
    # if DEBUG:
        # _view_operations(operations)

    if MODE == 'strong_simulation':
        sim_result = simulation_all_path(L, R, Mapping[-2], NUM_QUBITS, TYPE)
    elif MODE == 'weak_simulation':
        sim_result = weak_simulation(L, R, Mapping, NUM_QUBITS, TYPE, num_shots=1024)
    
    # sim_result = 0
    return sim_result

def dd_sparse_numpy(operations: List[GateOperation], INIT_MATRIX_SIZE: int, NUM_QUBITS: int, MODE: str = 'strong_simulation', DEBUG: bool = False):

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
        sim_result = dd_sparse(operations, INIT_MATRIX_SIZE, NUM_QUBITS, TYPE='numpy', MODE=MODE)

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


