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
    def __init__(self, L, R, leaf, NUM_QUBITS, TYPE):
        self.L = L
        self.R = R
        self.LL = self.L@self.L
        self.RR = self.R@self.R
        self.LR = self.L@self.R
        self.RL = self.R@self.L
        self.leaf = leaf
        self.NUM_QUBITS = NUM_QUBITS
        self.TYPE = TYPE
        self.sim_result = {}
        self.sim_time = 0

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
            
            # print(time.time() - now_time, "seconds elapsed for current iteration:", seq)


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
        self.sim_time = time.time() - start_simulation


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
    
    def simulation_time(self):
        return self.sim_time

def simulation_all_path(L, R, leaf, NUM_QUBITS, TYPE, START_TIME):
    sim = SimulationAllPath(L, R, leaf, NUM_QUBITS, TYPE)
    sim.simulation()
    # for val, seq in sim:
    #     pass  # Iterate through all paths to compute results
    return sim.results(), sim.simulation_time()

def show_matrix(matrix: List[List[complex]]):
    """
    Display a matrix in a readable format.
    """
    for row in matrix:
        print(" | ".join(f"{val:8.4f}" for val in row))
    print("\n")

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

class WeakSimulation:

    root_index = 0

    def __init__(self, 
                 L, 
                 R, 
                 Mapping: List[List[int]], 
                 NUM_QUBITS: int, 
                 TYPE: str = 'numpy', 
                 START_TIME: float = 0.0,
                 num_shots: int = 1024,
                 tol: float = 1e-10,
                 DEBUG: bool = False
    ):
        self.L = L
        self.R = R
        self.Mapping = Mapping
        self.NUM_QUBITS = NUM_QUBITS
        self.TYPE = TYPE
        self.START_TIME = START_TIME
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

    def verify_direction(self, Mapping):
        level = {}
        for lvl, nodes in enumerate(Mapping):
            for n in nodes:
                level[n] = lvl

        bad_edges = []
        for p, children in self.adj_list.items():
            for c in children:
                if level[p] >= level[c] and c not in Mapping[-2] and c not in Mapping[-1]:
                    bad_edges.append((p, c, level[p], level[c]))
        
        if len(bad_edges) > 0:
            print("[Error] Bad edges found in the adjacency list:")
            for p, c, lvl_p, lvl_c in bad_edges:
                print(f"Parent {p} (level {lvl_p}) -> Child {c} (level {lvl_c})")
            print("[Error] The adjacency list does not respect the mapping levels.")
        else:
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

        # self.verify_direction(self.Mapping)

        if self.DEBUG:
            print("[WEAK_SIM] adj_list:")
            show_adj_list(self.adj_list)

        # if self.DEBUG: print("[WEAK_SIM] Pruning Edges...")
        
        # self.prune_edges()
        
        # if self.DEBUG: print("[WEAK_SIM] Edges after pruning:")
        # if self.DEBUG: show_adj_list(self.adj_list)
        
        self.build_parent_map()
        
        if self.DEBUG: print("[WEAK_SIM] Parent Map:", self.parent_map)
        
        self.build_reachable_set()

        # self.check_loop()
        
        if self.DEBUG: print("[WEAK_SIM] Reachable Set:", self.reachable_set)
        
        # self.build_my_mapping()

        # self.check_two_mappings_are_equal()
        
        # print("[WEAK_SIM] Checking Mapping and Parent Map consistency...")
        # if self.check_mapping_and_parent_map(self.Mapping): 
        #     print("[Info] Mapping and Parent Map are consistent.")
        # else:
        #     print("[Error] Mapping and Parent Map are NOT consistent.")
            
        # print("[WEAK_SIM] Checking My Mapping and Parent Map consistency...")
        # if self.check_mapping_and_parent_map(self.my_Mapping):
        #     print("[Info] My Mapping and Parent Map are consistent.")
        # else:
        #     print("[Error] My Mapping and Parent Map are NOT consistent.")


        # self.filter_adj_list()
        # if self.DEBUG: print("[WEAK_SIM] Filtered Adjacency List:")
        # if self.DEBUG: show_adj_list(self.adj_list)

        # self.filter_parent_map()
        # if self.DEBUG: print("[WEAK_SIM] Filtered Parent Map:", self.parent_map)
        
        if self.DEBUG: print("[WEAK_SIM] Computing Downstream Contributions...")
        
        self.compute_downstream_contributions()
        
        if self.DEBUG: print("[WEAK_SIM] Downstream Contributions:", self.downstream)

        if self.DEBUG: print("[WEAK_SIM] Starting Simulation...")

        for _ in range(self.num_shots):
            if time.time() - self.START_TIME > 60:
                raise TimeoutError("Timeout during simulation: 60s.")

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

def weak_simulation(L, R, Mapping: List[List[int]], NUM_QUBITS: int, TYPE: str = 'numpy', START_TIME: float = 0.0, num_shots: int = 1024, tol: float = 1e-10):
    start_time = time.time()
    weak_sim = WeakSimulation(L, R, Mapping, NUM_QUBITS, TYPE, time.time(), num_shots, tol)
    return weak_sim.get_result(), time.time() - start_time


