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

from mbdd.gates import *
from mbdd.parser import *
from mbdd.simulator import *

def dd_sparse_old(operations: List[GateOperation], MATRIX_SIZE: int, NUM_QUBITS: int, TYPE: str, START_TIME: float, MODE: str = 'strong_simulation', DEBUG = False):
    L, R, Mapping = initialization_sparse(MATRIX_SIZE, NUM_QUBITS, TYPE)
    # print("Init Mapping: ", Mapping)
    # print("L:", L)
    # print("R:", R)

    if TYPE == 'numpy':
        import numpy as cp
        from scipy import sparse
    elif TYPE == 'cupy':
        import cupy as cp
        from cupyx.scipy import sparse
    
    for op in operations:
        if time.time() - START_TIME > 60:
            raise TimeoutError("Timeout in Circuit Operations: 60s.")
        
        MATRIX_SIZE = L.shape[0]
        # print("MATRIX_SIZE: ", MATRIX_SIZE)
        # print("L: ", L)
        # print("R: ", R)
        L_U, R_U = copy.deepcopy(L), copy.deepcopy(R)

        target_map = Mapping.getrow(op.target).indices.tolist()
        
        A = cp.ones(MATRIX_SIZE, dtype=complex)
        A[target_map] = op.a
        B = cp.zeros(MATRIX_SIZE, dtype=complex)
        B[target_map] = op.b
        C = cp.zeros(MATRIX_SIZE, dtype=complex)
        C[target_map] = op.c
        D = cp.ones(MATRIX_SIZE, dtype=complex)
        D[target_map] = op.d

        A = sparse.diags(A, offsets=0, format='csr')
        B = sparse.diags(B, offsets=0, format='csr')
        C = sparse.diags(C, offsets=0, format='csr')
        D = sparse.diags(D, offsets=0, format='csr')
     

        L_U = L @ A + R @ B
        R_U = L @ C + R @ D
        Mapping_U = Mapping.copy()

        if isinstance(op, SingleQubitGate):
            L, R = L_U, R_U
            # print("Mapping: ", Mapping)
        
        elif isinstance(op, ControlledGate):
            MATRIX_SIZE_OLD = MATRIX_SIZE
            control_up = [control for control in op.control if control < op.target]
            control_down = [control for control in op.control if control > op.target]

            # if len(control_up) != 0:
            # from itertools import chain
            # node_up = list(chain.from_iterable(Mapping.getrow(i).indices.tolist() for i in control_up))
            #     # L_up =

            
            # if len(control_down) != 0:


            for control in op.control:
                keep_U = cp.ones(MATRIX_SIZE, dtype=int)
                keep = cp.ones(MATRIX_SIZE_OLD, dtype=int)
                map_control = Mapping.getrow(control).indices.tolist()
                keep_U[map_control] = 0
                keep[map_control] = 0
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
                Mapping_U = sparse.hstack([Mapping_U, Mapping], format='csr')

            L, R, Mapping = L_U, R_U, Mapping_U
            # print("L: ", L)
            # print("R: ", R)
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
            # Mapping_U = [
            #     sorted(
            #         [n + MATRIX_SIZE_OLD * i
            #         for i in range(1, int(MATRIX_SIZE / MATRIX_SIZE_OLD) + 1)
            #         for n in mapping
            #         if n + MATRIX_SIZE_OLD * i in traverse] +
            #         [n for n in mapping if n in traverse]
            #     )
            #     for mapping in Mapping
            # ]
            # print("Mapping_U: ", Mapping_U)

            mapping_middle = time.time()
            # print("Mapping middle time spent:", mapping_middle - mapping_start)
            traverse_list = list(traverse)
            traverse_dict = {i: idx for idx, i in enumerate(traverse_list)}
            # Mapping = [[traverse_dict[node] for node in mapping_U] for mapping_U in Mapping_U]
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
                                    shape=(len(rows), MATRIX_SIZE), dtype=cp.float32)

            L = S @ L @ S.T
            R = S @ R @ S.T
            Mapping = Mapping @ S.T

            # print("end construct, time spent:", time.time() - construct_start)

            MATRIX_SIZE = L.shape[0]

        if not L.shape == R.shape:
            raise ValueError("Size of L and R not equal")
        
        if DEBUG:
            print("MATRIX_SIZE: ", MATRIX_SIZE)
            print("L: ", L)
            print("R: ", R)
            print("Mapping: ", Mapping)
    
    if DEBUG:
        view_operations(operations)

    exec_time = time.time() - START_TIME

    if MODE == 'strong_simulation':
        # print(Mapping.getrow(-2).indices.tolist())
        sim_result, sim_time = simulation_all_path(L, R, Mapping.getrow(-2).indices.tolist(), NUM_QUBITS, TYPE, START_TIME)
    elif MODE == 'weak_simulation':
        sim_result, sim_time = weak_simulation(L, R, Mapping.toarray(), NUM_QUBITS, TYPE, START_TIME, num_shots=1024)
    
    # sim_result = 0
    return exec_time, sim_result, sim_time

def dd_sparse(operations: List[GateOperation], MATRIX_SIZE: int, NUM_QUBITS: int, TYPE: str, START_TIME: float, MODE: str = 'strong_simulation', DEBUG = False):
    START_TIME = time.time()
    L, R, Mapping = initialization_sparse(MATRIX_SIZE, NUM_QUBITS, TYPE)
    # print("Init Mapping: ", Mapping)
    # print("L:", L)
    # print("R:", R)

    if TYPE == 'numpy':
        import numpy as cp
        from scipy import sparse
    elif TYPE == 'cupy':
        import cupy as cp
        from cupyx.scipy import sparse

    for op in operations:
        if time.time() - START_TIME > 60:
            raise TimeoutError("Timeout in Circuit Operations: 60s.")

        MATRIX_SIZE = L.shape[0]
        # print("MATRIX_SIZE: ", MATRIX_SIZE)
        # print("L: ", L)
        # print("R: ", R)
        # L_U, R_U = copy.deepcopy(L), copy.deepcopy(R)

        target_map = Mapping.getrow(op.target).indices.tolist()

        A = cp.ones(MATRIX_SIZE, dtype=cp.complex128)
        A[target_map] = op.a
        B = cp.zeros(MATRIX_SIZE, dtype=cp.complex128)
        B[target_map] = op.b
        C = cp.zeros(MATRIX_SIZE, dtype=cp.complex128)
        C[target_map] = op.c
        D = cp.ones(MATRIX_SIZE, dtype=cp.complex128)
        D[target_map] = op.d

        A = sparse.diags(A, offsets=0, format='csr')
        B = sparse.diags(B, offsets=0, format='csr')
        C = sparse.diags(C, offsets=0, format='csr')
        D = sparse.diags(D, offsets=0, format='csr')


        L_U = L @ A + R @ B
        R_U = L @ C + R @ D
        Mapping_U = Mapping.copy()
        # print("Mapping: ", Mapping)

        if isinstance(op, SingleQubitGate):
            L, R = L_U, R_U
            # print("Mapping: ", Mapping)

        elif isinstance(op, ControlledGate):
            # control_start = time.time()
            MATRIX_SIZE_OLD = MATRIX_SIZE

            if not isinstance(op.control, list):
                op.control = [op.control]

            # print("L: ", L)
            # print("R: ", R)

            # L_U = L_U @ sparse.diags((1-Mapping[op.control].sum(axis=0).toarray().ravel()), offsets=0, format='csr')
            # print("L_U before mul: ", L_U)
            L_U = L_U.multiply((cp.asarray(Mapping[op.control].sum(axis=0)).ravel() == 0).astype(L_U.dtype)).tocsr()
            # print("L_U after mul: ", L_U)

            # print("op.control: ", op, op.control)

            control_up = [control for control in op.control if control < op.target]
            control_down = [control for control in op.control if control > op.target]

            node_target = cp.asarray(Mapping[op.target].sum(axis=0)).ravel()
            # print("node_target: ", node_target)


            if len(control_up) != 0:
                node_up_all = cp.asarray(Mapping.sum(axis=0)).ravel() if op.target == NUM_QUBITS - 1 else cp.asarray(Mapping[[i for i in range(op.target+2)]].sum(axis=0)).ravel()
                # node_up_all = cp.asarray(Mapping.sum(axis=0)).ravel()
                # print("Mapping until t+2: ", Mapping[[i for i in range(op.target+2)]])
                node_up = cp.asarray(Mapping[[i for i in range(op.target)]].sum(axis=0)).ravel()
                node_control_up = cp.asarray(Mapping[control_up].sum(axis=0)).ravel()
                # print("node_up_all: ", node_up_all)
                # print("node_up: ", node_up)
                # print("node_control_up: ", node_control_up)
                node_up_not_zero = node_up_all != 0
                # node_up_not_zero = cp.flatnonzero(node_up_all != 0).astype(cp.int32, copy=False)
                L_up = L[node_up_not_zero][:, node_up_not_zero]
                R_up = R[node_up_not_zero][:, node_up_not_zero]

                # print("L_up: ", L_up)

                up_control = sparse.diags(node_control_up[:L_up.shape[0]], offsets=0, format='csr')
                up_target = sparse.diags(node_target[:L_up.shape[0]], offsets=0, format='csr')
                up = sparse.diags(node_up[:L_up.shape[0]], offsets=0, format='csr')

                # print("up_control: ", up_control)
                # print("up_target: ", up_target)
                # print("up: ", up)

                L_up_control = L_up @ up_control
                L_up_target = L_up @ up_target
                R_up_target = R_up @ up_target
                L_up = L_up @ up
                R_up = R_up @ up

                # print("L_U: ", L_U)

                # print("L_up_control: ", L_up_control)
                # print("L_up_target: ", L_up_target)
                # print("R_up_target: ", R_up_target)


                # M_U = sparse.diags((1-node_control_up), offsets=0, format='csr')
                # print("M_U: ", M_U)
                # L_U = L_U @ M_U

                # print("L_U.shape: ", L_U.shape)
                # print("L_up.shape: ", L_up.shape)
                # print("L_up_target.shape: ", L_up_target.shape)
                # print("L_up_control.shape: ", L_up_control.shape)
                # print("R_U.shape: ", R_U.shape)
                # print("R_up.shape: ", R_up.shape)
                # print("R_up_target.shape: ", R_up_target.shape)

                # print("L_U: ", L_U)
                # print("R_U: ", R_U)
                # print("L_U[:L_up.shape[0], :L_up.shape[0]]: ", L_U[:L_up.shape[0], :L_up.shape[0]])
                # print("L_U[:L_up.shape[0], L_up.shape[0]:]: ", L_U[:L_up.shape[0], L_up.shape[0]:])
                # print("L_U[L_up.shape[0]:, :L_up.shape[0]]: ", L_U[L_up.shape[0]:, :L_up.shape[0]])
                # print("L_U[L_up.shape[0]:, L_up.shape[0]:]: ", L_U[L_up.shape[0]:, L_up.shape[0]:])
                # print("L_up_target: ", L_up_target)
                # print("L_up_control: ", L_up_control)
                # print("L_up: ", L_up)


                L_U = sparse.bmat([[L_U[:L_up.shape[0], :L_up.shape[0]], L_U[:L_up.shape[0], L_up.shape[0]:], L_up_target],
                                [L_U[L_up.shape[0]:, :L_up.shape[0]], L_U[L_up.shape[0]:, L_up.shape[0]:], None],
                                [L_up_control, None, L_up]], format='csr')
                # print("L_U PASS")
                R_U = sparse.bmat([[R_U[:R_up.shape[0], :], R_up_target],
                                [R_U[R_up.shape[0]:, :], None],
                                [None, R_up]], format='csr')
                # print("R_U PASS")

                # print("L_U: ", L_U)
                # print("R_U: ", R_U)

                # print("L_U.shape: ", L_U.shape)

                # print("cpuMappingup: ", Mapping)
                # print("Mapping[:, node_up_all]: ", Mapping[:, node_up_all])
                # print("Mapping[node_up, :]: ", Mapping[node_up, :])

                # print("cp.array(node_up_all, dtype=bool): ", cp.array(node_up_all, dtype=bool))
                # print("cp.where(cp.array(node_up_all, dtype=bool))[0]: ", cp.where(cp.array(node_up_all, dtype=bool))[0])
                # print("Mapping[:, cp.where(cp.array(node_up_all, dtype=bool))[0]]: ", Mapping[:, cp.where(cp.array(node_up_all, dtype=bool))[0]])
                # print("Mapping[:, cp.array(node_up_all, dtype=bool)]: ", Mapping[:, cp.array(node_up_all, dtype=bool)])

                mask = cp.asarray(node_up_all, dtype=bool)
                idx  = cp.flatnonzero(mask).astype(cp.int32)

                S_up = sparse.csr_matrix(
                    (cp.ones(idx.size, dtype=cp.complex128),
                    (idx, cp.arange(idx.size, dtype=cp.int32))),
                    shape=(Mapping.shape[1], int(idx.size))
                )

                Mapping_U = sparse.hstack([Mapping, Mapping@S_up], format='csr')
                # Mapping_U = sparse.hstack([Mapping, Mapping], format='csr') if op.target == NUM_QUBITS - 1 else sparse.hstack([Mapping[:, :op.target+2], Mapping[:, op.target+1:]], format='csr')
                # print("Mapping_U.shape: ", Mapping_U.shape)
                # print("Mapping_U: ", Mapping_U)

            # print("Mapping: ", Mapping)
            # print("Mapping[:op.target+1]: ", Mapping[:, :op.target+1])
            # print("Mapping[op.target:]: ", Mapping[:, op.target:])

            if len(control_down) != 0:
                node_down = cp.asarray(Mapping[[i for i in range(control_down[-1])]].sum(axis=0)).ravel()
                node_down_all = cp.asarray(Mapping[[i for i in range(Mapping.shape[0])]].sum(axis=0)).ravel() if control_down[-1] == NUM_QUBITS - 1 else cp.asarray(Mapping[[i for i in range(control_down[-1]+2)]].sum(axis=0)).ravel()
                node_control_down = cp.asarray(Mapping[control_down].sum(axis=0)).ravel()

                node_down_all_not_zero = node_down_all != 0
                L_down = L[node_down_all_not_zero][:, node_down_all_not_zero]
                R_down = R[node_down_all_not_zero][:, node_down_all_not_zero]
                # print("L_down: ", L_down)
                # print("R_down: ", R_down)


                down = sparse.diags(node_down[:L_down.shape[0]], offsets=0, format='csr')

                node_control_down_except_last = cp.asarray(Mapping[control_down[:-1]].sum(axis=0)).ravel()
                node_control_down_only_last = cp.asarray(Mapping[control_down[-1]].sum(axis=0)).ravel()
                # print("node_down: ", node_down)
                # print("node_down_all: ", node_down_all)
                # print("node_control_down: ", node_control_down)
                # print("node_control_down_except_last: ", node_control_down_except_last)
                # print("node_control_down_only_last: ", node_control_down_only_last)

                down_last_control = sparse.diags(node_control_down_only_last[:L_down.shape[0]], offsets=0, format='csr')
                L_down_last_control = L_down @ down_last_control

                R_down_last_control = sparse.csr_matrix(R_down.shape, dtype=cp.complex128)
                R_down_last_control_real = R_down @ down_last_control

                L_down = L_down @ down
                R_down = R_down @ down

                down_other_control_neg = sparse.diags((1-node_control_down_except_last[:L_down.shape[0]]), offsets=0, format='csr')
                L_down_all = L_down @ down_other_control_neg
                R_down_all = R_down.copy()

                mask = cp.asarray(node_down_all, dtype=bool)
                idx  = cp.flatnonzero(mask).astype(cp.int32)

                S_down = sparse.csr_matrix(
                    (cp.ones(idx.size, dtype=cp.complex128),
                    (idx, cp.arange(idx.size, dtype=cp.int32))),
                    shape=(Mapping.shape[1], int(idx.size))
                )
                # print("node_down_all: ", node_down_all)
                # print("Mapping_U: ", Mapping_U)
                # print("S_down: ", S_down)

                Mapping_U = sparse.hstack([Mapping_U, Mapping@S_down], format='csr')

                if len(control_down) > 1:
                    L_down_control_go_left = L_down @ sparse.diags(node_control_down_except_last[:L_down.shape[0]], offsets=0, format='csr')
                    # print("L_down_control_go_left: ", L_down_control_go_left)
                    # print("L_down_all.shape: ", L_down_all.shape)
                    # print("L_down.shape: ", L_down.shape)
                    # print("L_down_control_go_left.shape: ", L_down_control_go_left.shape)
                    L_down_all = sparse.bmat([[L_down_all, None],
                                        [L_down_control_go_left, L_down]], format='csr')
                    R_down_all = sparse.bmat([[R_down_all, None],
                                        [None, R_down]], format='csr')

                    # print("L_down_all: ", L_down_all)
                    # print("R_down_all: ", R_down_all)
                    L_down_last_control = sparse.hstack([L_down_last_control, L_down_last_control], format='csr')
                    R_down_last_control = sparse.hstack([R_down_last_control, R_down_last_control_real], format='csr')
                    Mapping_U = sparse.hstack([Mapping_U, Mapping@S_down], format='csr')

                down_target = sparse.diags(node_target[:L_down.shape[0]], offsets=0, format='csr')
                L_down_target = L_down @ down_target
                R_down_target = R_down @ down_target
                # print("node_target: ", node_target)
                # print("L_down: ", L_down)
                # print("down_target: ", down_target)
                # print("L_down_target: ", L_down_target)
                # print("R_down_target: ", R_down_target)

                if len(control_up) == 0:
                    # print("L_U[:L_down.shape[0], :-L_down.shape[0]]: ", L_U[:L_down.shape[0], :-L_down.shape[0]])
                    # print("L_U[:L_down.shape[0], -L_down.shape[0]:]: ", L_U[:L_down.shape[0], -L_down.shape[0]:])
                    
                    # print("L_U: ", L_U)
                    L_U = sparse.bmat([[L_U[:L_down.shape[0], :L_down.shape[0]], L_U[:L_down.shape[0], L_down.shape[0]:], L_down_last_control],
                                    [L_U[L_down.shape[0]:, :L_down.shape[0]], L_U[L_down.shape[0]:, L_down.shape[0]:], None],
                                    [L_down_target, None, L_down_all[:L_down.shape[0]]],
                                    [None, None, L_down_all[L_down.shape[0]:]]], format='csr')
                    R_U = sparse.bmat([[R_U[:R_down.shape[0], :R_down.shape[0]], R_U[:R_down.shape[0], R_down.shape[0]:], R_down_last_control],
                                    [R_U[R_down.shape[0]:, :R_down.shape[0]], R_U[R_down.shape[0]:, R_down.shape[0]:], None],
                                    [R_down_target, None, R_down_all[:R_down.shape[0]]],
                                    [None, None, R_down_all[R_down.shape[0]:]]], format='csr')

                    # print("L_U: ", L_U)
                    # print("R_U: ", R_U)
                else:
                    # print("MATRIX_SIZE: ", MATRIX_SIZE)
                    # print("L_U: ", L_U)
                    # print("L_down: ", L_down)
                    # print("L_down_last_control: ", L_down_last_control)
                    # print("L_down_target: ", L_down_target)
                    # print("L_down_all: ", L_down_all)
                    
                    # Should be correct now...FINALLY

                    L_U = sparse.bmat([[L_U[:L_down.shape[0], :L_down.shape[0]], L_U[:L_down.shape[0], L_down.shape[0]:MATRIX_SIZE], L_U[:L_down.shape[0], MATRIX_SIZE:], L_down_last_control],
                                    [L_U[L_down.shape[0]:, :L_down.shape[0]], L_U[L_down.shape[0]:, L_down.shape[0]:MATRIX_SIZE], L_U[L_down.shape[0]:, MATRIX_SIZE:], None],
                                    [L_down_target, None, L_down_target[:, :L_U.shape[0]-MATRIX_SIZE], L_down_all[:L_down.shape[0]]],
                                    [None, None, None, L_down_all[L_down.shape[0]:]]], format='csr')
                    R_U = sparse.bmat([[R_U[:R_down.shape[0], :R_down.shape[0]], R_U[:R_down.shape[0], R_down.shape[0]:MATRIX_SIZE], R_U[:R_down.shape[0], MATRIX_SIZE:], R_down_last_control],
                                    [R_U[R_down.shape[0]:, :R_down.shape[0]], R_U[R_down.shape[0]:, R_down.shape[0]:MATRIX_SIZE], R_U[R_down.shape[0]:, MATRIX_SIZE:], None],
                                    [R_down_target, None, R_down_target[:, :R_U.shape[0]-MATRIX_SIZE], R_down_all[:R_down.shape[0]]],
                                    [None, None, None, R_down_all[R_down.shape[0]:]]], format='csr')
            
            L, R, Mapping, MATRIX_SIZE = L_U, R_U, Mapping_U, L_U.shape[0]
            # print("L: ", L)
            # print("R: ", R)
            # print("start traverse")
            if DEBUG:
                print("L: ", L)
                print("R: ", R)
            # print("end control, time spent:", time.time() - control_start)
            # traverse_start = time.time()

            # L_B = L.astype(bool).astype(float)
            # R_B = R.astype(bool).astype(float)

            # LR = L_B + R_B
            # LR = LR.astype(bool)
            # LR = LR.astype(float)

            LR_bool = (L != 0).maximum(R != 0)
            LR = LR_bool.astype(float)

            # traverse_time_one = time.time()
            # print("traverse_time_one time spent:", traverse_time_one - traverse_start)

            reach_td = sparse.csr_matrix((MATRIX_SIZE, 1), dtype=bool)
            reach_td[0, 0] = True
            prev = -1
            while reach_td.nnz != prev:
                prev = reach_td.nnz
                reach_td = (reach_td + (LR @ reach_td)).astype(bool)

                # print("Traverse: ", traverse)

            # print("reach_td: ", reach_td)
            # traverse_time_two = time.time()
            # print("traverse_time_two time spent:", traverse_time_two - traverse_time_one)

            reach_bu = sparse.csr_matrix((MATRIX_SIZE, 1), dtype=bool)
            reach_bu[Mapping.indices[Mapping.indptr[NUM_QUBITS]], 0] = True
            reach_bu[Mapping.indices[Mapping.indptr[NUM_QUBITS + 1]], 0] = True
            prev = -1
            while reach_bu.nnz != prev:
                prev = reach_bu.nnz
                reach_bu = (reach_bu + (LR.T @ reach_bu)).astype(bool)
            
            # print("reach_bu: ", reach_bu)
            # traverse_time_three = time.time()
            # print("traverse_time_three time spent:", traverse_time_three - traverse_time_two)

            traverse = set(reach_td.multiply(reach_bu).toarray().ravel().nonzero()[0].tolist())
        
            # print("Traverse: ", traverse)
            # traverse_end = time.time()

            # print("traverse_time_four spent:", traverse_end - traverse_time_three)
            # print("end traverse, time spent:", traverse_end - traverse_start)

            # print("start mapping")
            # mapping_start = time.time()
            # Mapping_U = [
            #     sorted(
            #         [n + MATRIX_SIZE_OLD * i
            #         for i in range(1, int(MATRIX_SIZE / MATRIX_SIZE_OLD) + 1)
            #         for n in mapping
            #         if n + MATRIX_SIZE_OLD * i in traverse] +
            #         [n for n in mapping if n in traverse]
            #     )
            #     for mapping in Mapping
            # ]
            # print("Mapping_U: ", Mapping_U)

            # mapping_middle = time.time()
            # print("Mapping middle time spent:", mapping_middle - mapping_start)
            traverse_list = list(traverse)
            traverse_dict = {i: idx for idx, i in enumerate(traverse_list)}
            # Mapping = [[traverse_dict[node] for node in mapping_U] for mapping_U in Mapping_U]
            # print("Mapping_U after dict: ", Mapping_U)
            # Mapping = copy.deepcopy(Mapping_U)
            # print("end mapping, time spent:", time.time() - mapping_middle)

            # print("start construct")
            # construct_start = time.time()


            not_traverse = cp.array(list(set(range(MATRIX_SIZE)) - traverse))

            rows = cp.arange(MATRIX_SIZE)
            rows = rows[~cp.isin(rows, not_traverse)]
            one = cp.ones_like(rows, dtype=cp.int32)


            S = sparse.csr_matrix((one, (cp.arange(len(rows)), rows)),
                                    shape=(len(rows), MATRIX_SIZE), dtype=cp.complex128)

            L = S @ L @ S.T
            R = S @ R @ S.T

            # print("S.shape: ", S.shape)
            # print("Mapping.shape: ", Mapping.shape)
            # print("op: ", op)
            # print("Mapping before select: ", Mapping)
            # print("Select: ", S)
            Mapping = Mapping @ S.T
            # print("Mapping after select: ", Mapping)

            # print("End Mapping.shape: ", Mapping.shape)

            # m = Mapping.shape[1]
            # Mapping = Mapping.tocsc()[:, cp.arange(m, dtype=cp.int32)].tocsr()

            # print("op: ", op)
            # print("End Mapping: ", Mapping)
            # print("L: ", L)
            # print("R: ", R)

            # print("end construct, time spent:", time.time() - construct_start)

            if Mapping[-2].nnz == 0:
                print("op: ", op)
                print("Mapping: ", Mapping)

            MATRIX_SIZE = L.shape[0]

            # reorder_start = time.time()

            reorder_mapping = Mapping.indices.tolist()

            L = L[reorder_mapping, :][:, reorder_mapping]

            R = R[reorder_mapping, :][:, reorder_mapping]

            Mapping = Mapping[:, reorder_mapping]

            # print("end reorder, time spent:", time.time() - reorder_start)

            # for i in range(MATRIX_SIZE-3):
            #     for j in range(i+1, MATRIX_SIZE-2):
            #         if (L.tocsc()[:, i] - L.tocsc()[:, j]).nnz == 0 and (R.tocsc()[:, i] - R.tocsc()[:, j]).nnz == 0:
            #             print("op: ", op)
            #             print("(i, j, size): ", (i, j, MATRIX_SIZE))
            #             print(f"L[:, {i}]: ", L[:, i].toarray().ravel())
            #             print(f"L[:, {j}]: ", L[:, j].toarray().ravel())

            #             print(f"R[:, {i}]: ", R[:, i].toarray().ravel())
            #             print(f"R[:, {j}]: ", R[:, j].toarray().ravel())


        # finish_Mapping_reorder = time.time()

        # print("reorder time: ", finish_mapping_reorder_list - start_reorder, finish_L_reorder - finish_mapping_reorder_list, finish_R_reorder - finish_L_reorder, finish_Mapping_reorder - finish_R_reorder)

        # print("reorder mapping: ", reorder_mapping)
        # print("After reorder L: ", L)
        # print("After reorder R: ", R)
        # print("After reorder Mapping: ", Mapping)


        if not L.shape == R.shape:
            raise ValueError("Size of L and R not equal")

        if DEBUG:
            print("MATRIX_SIZE: ", MATRIX_SIZE)
            print("L: ", L)
            print("R: ", R)
            print("Mapping: ", Mapping)

    if DEBUG:
        _view_operations(operations)

    exec_time = time.time() - START_TIME

    # print("MATRIX_SIZE: ", MATRIX_SIZE)
    if MODE == 'strong_simulation':
        # print(MATRIX_SIZE, Mapping.getrow(-2).indices.tolist())
        sim_result, sim_time = simulation_all_path(L, R, Mapping.getrow(-2).indices.tolist(), NUM_QUBITS, TYPE, START_TIME)
    elif MODE == 'weak_simulation':
        sim_result, sim_time = weak_simulation(L, R, Mapping.toarray(), NUM_QUBITS, TYPE, START_TIME, num_shots=1024)

    # sim_result = 0
    return exec_time, sim_result, sim_time, MATRIX_SIZE

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
        exec_time, sim_result, sim_time, matrix_size = dd_sparse(operations, INIT_MATRIX_SIZE, NUM_QUBITS, TYPE='numpy', START_TIME=np_start, MODE=MODE)

        np_end = time.time()

        stop_event.set()
        monitor_thread.join()

        result_dict["exec_time"] = exec_time
        result_dict["sim_time"] = sim_time
        result_dict["time_spent"] = np_end - np_start
        result_dict["matrix_size"] = matrix_size
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
        result_dict["exec_time"] = None
        result_dict["sim_time"] = None
        result_dict['matrix_size'] = None
        traceback.print_exc()

    # print("Time spent: ", np_end - np_start)
    return result_dict

def dd_sparse_cupy(operations: List[GateOperation], INIT_MATRIX_SIZE: int, NUM_QUBITS: int, MODE: str = 'strong_simulation', DEBUG: bool = False):
    try:
        nvmlInit()
        gpu_available = True
        handle = nvmlDeviceGetHandleByIndex(0)
    except:
        gpu_available = False

    def monitor_usage(stop_event, result_dict, interval=0.5):
        pid = os.getpid()
        process = psutil.Process(pid)
        cpu_usages = []
        mem_usages = []
        gpu_usages = []
        gpu_mems = []
        
        while not stop_event.is_set():
            cpu = process.cpu_percent(interval=None)
            mem = process.memory_info().rss / 1024**2  # MB
            cpu_usages.append(cpu)
            mem_usages.append(mem)

            if gpu_available:
                try:
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    gpu_usages.append(util.gpu)
                    gpu_mems.append(mem_info.used / 1024**2)
                except:
                    gpu_usages.append(0)
                    gpu_mems.append(0)

            time.sleep(interval)

        result_dict["avg_cpu"] = sum(cpu_usages)/len(cpu_usages) if cpu_usages else 0.0
        result_dict["peak_mem"] = max(mem_usages) if mem_usages else 0.0

        if gpu_usages:
            result_dict["avg_gpu"] = sum(gpu_usages)/len(gpu_usages)
            result_dict["peak_gpu_mem"] = max(gpu_mems)
        else:
            result_dict["avg_gpu"] = 0.0
            result_dict["peak_gpu_mem"] = 0.0

    import cupy as cp
    import cupyx.scipy.sparse as sparse
    try:
        result_dict = {}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_usage, args=(stop_event, result_dict))
    
        cp_start = time.time()
        monitor_thread.start()
        # for i in range(1024):
        #     if time.time() - cp_start > 60:
        #         raise TimeoutError("Simulation Timeout: 60s.")
        exec_time, sim_result, sim_time, matrix_size = dd_sparse(operations, INIT_MATRIX_SIZE, NUM_QUBITS, TYPE='cupy', START_TIME=cp_start, MODE=MODE, DEBUG=DEBUG)

        cp_end = time.time()
        stop_event.set()
        monitor_thread.join()
        # print("|0>*n: ", compute_result(L, R, [0]*NUM_QUBITS, Mapping[-2]))
        # print("|1>*n: ", compute_result(L, R, [1]*NUM_QUBITS, Mapping[-2]))

        device_id = cp.cuda.runtime.getDevice()
        device_props = cp.cuda.runtime.getDeviceProperties(device_id)
        # print(f"Device {device_id} name:", device_props['name'].decode())
        # print(sim_result)
        # for comparison function
        result_dict["sim_result"] = sim_result

        result_dict["exec_time"] = exec_time
        result_dict["sim_time"] = sim_time
        result_dict["time_spent"] = cp_end - cp_start
        result_dict["matrix_size"] = matrix_size
        result_dict["error"] = None

    except Exception as e:
        cp_end = time.time()
        stop_event.set()
        monitor_thread.join()
        # print("Error in cupy computation:", e)
        result_dict["time_spent"] = cp_end - cp_start
        result_dict["error"] = str(e)
        result_dict["exec_time"] = None
        result_dict["sim_time"] = None
        result_dict['matrix_size'] = None
        traceback.print_exc()


    # print("Time spent: ", cp_end - cp_start)
    return result_dict

def cudaq_sv(path: str, NUM_QUBITS, MODE: str = 'nvidia'):
    # num_shots = 2**max(NUM_QUBITS, 10)
    num_shots = 1024
    # output_path = Path(path + 'circuit_cudaq.py')
    # !npx q-convert -i {input_path} -s qasm -o {output_path} -d cudaq

    method_style_gates = {"x", "y", "z", "h", "rx", "ry", "rz", "t", "s", "swap"}

    def parse_qasm_to_cudaq_python(qasm_file: str, output_py_file: str):
        with open(qasm_file, 'r') as f:
            lines = f.readlines()

        cudaq_lines = [
            "import cudaq\n",
            f"cudaq.set_target('{MODE}')\n",
            "@cudaq.kernel\n",
            "def circuit():\n"
        ]
        indent = "    "

        for line in lines:
            line = line.strip().rstrip(';')
            if not line or line.startswith("OPENQASM") or line.startswith("include") or line.startswith("//"):
                continue
            ## 
            if match := re.match(r"qreg\s+(\w+)\[(\d+)\]", line):
                regname, size = match.groups()
                cudaq_lines.append(f"{indent}{regname} = cudaq.qvector({size})\n")
                continue

            if match := re.match(r'([a-zA-Z_][\w]*)(\([^)]*\))?\s+(.*)', line):
                gate_raw, param_group, arglist = match.groups()
                args = [arg.strip() for arg in arglist.split(",") if arg.strip()]

                ctrl_count = len(gate_raw) - len(gate_raw.lstrip("c"))
                base_gate = gate_raw[ctrl_count:]

                params = []
                if param_group:                       # "(theta,phi,…)" → list[str]
                    params = [p.strip() for p in param_group[1:-1].split(',')]
                    params = [_eval_expr(p) for p in params]

                if ctrl_count > 0:
                    controls = args[:ctrl_count]
                    targets = args[ctrl_count:]

                    if base_gate in method_style_gates and len(targets) == 1:
                        ctrl_list = f"[{', '.join(controls)}]"
                        cudaq_lines.append(
                            f"{indent}{base_gate}.ctrl({ctrl_list}, {targets[0]})\n"
                        )
                    else:
                        ctrl_tuple = ", ".join(controls)
                        lambda_call = f"{base_gate}({', '.join(targets)})"
                        # cudaq_lines.append(
                        #     f"{indent}cudaq.control(({ctrl_tuple}), lambda: {lambda_call})\n"
                        # )
                else:
                    if base_gate in method_style_gates:
                        if not params:
                            cudaq_lines.append(f"{indent}{base_gate}({', '.join(args)})\n")
                        else:
                            if base_gate in {"rx", "ry", "rz"} and len(params) == 1:
                                cudaq_lines.append(f"{indent}{base_gate}({params[0]}, {', '.join(args)})\n")
                            

                    else:
                        if base_gate in {"u1", "u2", "u3"}:
                                if base_gate == "u1" and len(params) == 1:
                                    cudaq_lines.append(f"{indent}u3(0,0,{params[0]}, {', '.join(args)})\n")
                                if base_gate == "u2" and len(params) == 2:
                                    cudaq_lines.append(f"{indent}u3({_eval_expr(str(math.pi/2))},{params[0]},{params[1]}, {', '.join(args)})\n")
                                if base_gate == "u3" and len(params) == 3:
                                    cudaq_lines.append(f"{indent}u3({params[0]},{params[1]},{params[2]}, {', '.join(args)})\n")
                        if base_gate == "p" and len(params) == 1:
                            cudaq_lines.append(f"{indent}rz({params[0]}, {', '.join(args)})\n")
                        if base_gate == "sdg":
                            cudaq_lines.append(f"{indent}rz({_eval_expr(str(math.pi/-2))}, {', '.join(args)})\n")
                        if base_gate == "tdg":
                            cudaq_lines.append(f"{indent}rz({_eval_expr(str(math.pi/-4))}, {', '.join(args)})\n")
                        if base_gate == "sx":
                            cudaq_lines.append(f"{indent}rx({_eval_expr(str(math.pi/2))}, {', '.join(args)})\n")
                        if base_gate == "sxdg":
                            cudaq_lines.append(f"{indent}rx({_eval_expr(str(math.pi/-2))}, {', '.join(args)})\n")

        cudaq_lines.append(f"\ncounts = cudaq.sample(circuit, shots_count={num_shots})\nprint(counts)")
        with open(output_py_file, 'w') as f:
            f.writelines(cudaq_lines)

    def parse_real_to_cudaq(real_file: str, output_py_file: str):
        with open(real_file, 'r') as f:
            lines = f.readlines()

            cudaq_lines = [
                "import cudaq\n",
                f"cudaq.set_target('nvidia')\n",
                "@cudaq.kernel\n",
                "def circuit():\n"
            ]
            indent = "    "

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(".version") or line.startswith(".begin"):
                    continue
                ## 
                if line.startswith(".numvars"):
                    num_qubits = int(line.split()[1])
                    cudaq_lines.append(f"{indent}q = cudaq.qvector({num_qubits})\n")
                    continue
                # Handle .constants with only one argument (e.g., ".constants 0----")
                if line.startswith(".constants") and len(line.split()) == 2 :
                    # Example: .constants 0----
                    value = line.split()[1]
                    # For .constants like "0----", apply X to each qubit where char is '1'
                    for idx, bit in enumerate(value):
                        if bit == '1':
                            cudaq_lines.append(f"{indent}x(q[{idx}])\n")
                if line.startswith(".garbage"):
                        # .garbage 1111-  (1: don't measure, -: measure)
                        garbage_bits = line.split()[1]
                        measure_list = []
                        for idx, bit in enumerate(garbage_bits):
                            if bit == '-':
                                measure_list.append(idx)
                        # Optionally, you can use measure_list later for measurement logic

                # Parse gate lines, e.g., "t2 q[2] q[1]"
                if m := re.match(r'^([a-zA-Z\+]+)(\d*)\s+((?:q\[\d+\]\s*)+)', line):
                    
                    gate_type= m.group(1)
                    num_qubits = int(m.group(2)) if m.group(2) else 1
                    qubits = m.group(3).split()
                    if len(qubits) != num_qubits:
                        raise ValueError(f"Gate {gate_type_with_num} expects {num_qubits} qubits, got {len(qubits)}")
                    target = qubits[-1]
                    controls = qubits[:-1]
                    ctrl_list = f"[{', '.join(controls)}]"
                    # Map gate type to cudaq syntax
                    if gate_type.lower() == 't' :
                        #multi-control X
                        if len(controls) > 0:
                            cudaq_lines.append(f"{indent}x.ctrl({ctrl_list}, {target})\n")
                        else:
                            cudaq_lines.append(f"{indent}x({target})\n")
                    elif gate_type.lower() == 'f' and len(controls) > 0:
                        if len(controls) > 0:
                            cudaq_lines.append(f"{indent}x.ctrl({target}, {controls[-1]})\n")
                            cudaq_lines.append(f"{indent}x.ctrl({ctrl_list}, {target})\n")
                            cudaq_lines.append(f"{indent}x.ctrl({target}, {controls[-1]})\n")
                        else:
                            cudaq_lines.append(f"{indent}x({target})\n")
                    elif gate_type.lower() == 'v':
                        if len(controls) > 0:
                            cudaq_lines.append(f"{indent}cudaq.ctrl({ctrl_list}, lambda: cudaq.rx({_eval_expr("pi/2")}, {target}))\n")
                        else:
                            cudaq_lines.append(f"{indent}cudaq.rx({_eval_expr("pi/2")}, {target})\n")
                    elif gate_type.lower() == 'p':
                        if len(controls) > 0:
                            cudaq_lines.append(f"{indent}x.ctrl({ctrl_list}, {target})\n")
                            new_controls = controls[:-1]
                            new_ctrl_list = f"[{', '.join(new_controls)}]" 
                            cudaq_lines.append(f"{indent}x.ctrl({new_ctrl_list}, {controls[-1]})\n")
                        else:
                            continue
                    elif gate_type.lower() == 'pi':
                        if len(controls) > 0:
                            new_controls = controls[:-1]
                            new_ctrl_list = f"[{', '.join(new_controls)}]" 
                            cudaq_lines.append(f"{indent}x.ctrl({new_ctrl_list}, {controls[-1]})\n")
                            cudaq_lines.append(f"{indent}x.ctrl({ctrl_list}, {target})\n")
                        else:
                            continue
                    elif gate_type.lower() == 'v+':
                        if len(controls) > 0:
                            cudaq_lines.append(f"{indent}cudaq.ctrl({ctrl_list}, lambda: cudaq.rx({_eval_expr("-pi/2")}, {target}))\n")
                        else:
                            cudaq_lines.append(f"{indent}cudaq.rx({_eval_expr("-pi/2")}, {target})\n")
                    # Add more gate mappings as needed
                    else:
                        continue

        cudaq_lines.append(f"\ncounts = cudaq.sample(circuit, shots_count={num_shots})\nprint(counts)")
        with open(output_py_file, 'w') as f:
            f.writelines(cudaq_lines)

    path_parent = str(Path(path).parent)

    # print(path_parent)
    output_path = Path(path_parent + '/' + str(path.stem) + '_' + str(MODE) + '.py')
    print("output path: ", output_path)
    #TODO
    parse_qasm_to_cudaq_python(path, output_path)

    try:
        nvmlInit()
        gpu_available = True
        handle = nvmlDeviceGetHandleByIndex(0)
    except:
        gpu_available = False

    def monitor_usage(stop_event, result_dict, interval=0.5):
        pid = os.getpid()
        process = psutil.Process(pid)
        cpu_usages = []
        mem_usages = []
        gpu_usages = []
        gpu_mems = []

        while not stop_event.is_set():
            cpu = process.cpu_percent(interval=None)
            mem = process.memory_info().rss / 1024**2  # MB
            cpu_usages.append(cpu)
            mem_usages.append(mem)

            if gpu_available:
                try:
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    gpu_usages.append(util.gpu)
                    gpu_mems.append(mem_info.used / 1024**2)
                except:
                    gpu_usages.append(0)
                    gpu_mems.append(0)

            time.sleep(interval)

        result_dict["avg_cpu"] = sum(cpu_usages)/len(cpu_usages) if cpu_usages else 0.0
        result_dict["peak_mem"] = max(mem_usages) if mem_usages else 0.0

        if gpu_usages:
            result_dict["avg_gpu"] = sum(gpu_usages)/len(gpu_usages)
            result_dict["peak_gpu_mem"] = max(gpu_mems)
        else:
            result_dict["avg_gpu"] = 0.0
            result_dict["peak_gpu_mem"] = 0.0

    import subprocess

    result_dict = {}
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_usage, args=(stop_event, result_dict))
    timeout_secs = 60

    cudaq_start = time.time()
    monitor_thread.start()

    try:
        result = subprocess.run(
            ['python', output_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_secs
        )
        cudaq_end = time.time()
        result_dict['sim_result'] = result.stdout
        stop_event.set()
        monitor_thread.join()
        result_dict["time_spent"] = cudaq_end - cudaq_start
        result_dict['error'] = None

    except subprocess.TimeoutExpired as e:
        cudaq_end = time.time()
        stop_event.set()
        monitor_thread.join()
        result_dict["time_spent"] = cudaq_end - cudaq_start
        result_dict['sim_result'] = str(e.stderr)
        result_dict['error'] = f"Timeout after {timeout_secs}s"
        print(f"Timed out after {timeout_secs}s")


    except subprocess.CalledProcessError as e:
        cudaq_end = time.time()
        stop_event.set()
        monitor_thread.join()
        result_dict["time_spent"] = cudaq_end - cudaq_start
        result_dict['sim_result'] = str(e.stderr)
        result_dict['error'] = str(e.stderr)
        print("Error output:\n", e.stderr)
        traceback.print_exc()


    return result_dict