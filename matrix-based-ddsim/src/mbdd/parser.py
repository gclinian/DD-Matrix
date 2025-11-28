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


_PI_ENV = {"pi": math.pi, "tau": 2*math.pi, "e": math.e}

def eval_expr(expr: str) -> float:
    return eval(expr, {"__builtins__": None}, _PI_ENV)

def rot_x(theta: float):
    c, s = math.cos(theta/2), math.sin(theta/2)
    return [[c, -1j*s], [-1j*s, c]]

def rot_y(theta: float):
    c, s = math.cos(theta/2), math.sin(theta/2)
    return [[c, -s], [s, c]]

def rot_z(theta: float):
    return [[cmath.exp(-1j*theta/2), 0],
            [0, cmath.exp(1j*theta/2)]]

def phase(lambda_):
    return [[1, 0], [0, cmath.exp(1j*lambda_)]]

def extract_qubit_index(s: str) -> int:
    """Extract qubit index from string like 'q[0]' -> 0"""
    match = re.search(r'\[(\d+)\]', s)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid qubit format: {s}")
## .reals file parser for revlib
def parse_reals(path: Path) -> Tuple[int, List[GateOperation]]:
    gate_str_to_enum = { 'v' : GateType.V, 'v+': GateType.V_plus}
    num_qubits, gate_list = None, []
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip().rstrip('\n')
            if not line or line.startswith('#'):
                continue
            # version ,varialbe, inputs, outputs, constants, garbage line (ignore)
            if line.startswith('.version') or line.startswith('.variable') or line.startswith('.inputs') or line.startswith('.outputs') or line.startswith('.constants') or line.startswith('.garbage') or line.startswith('.begin') or line.startswith('.end'):
                continue
            # numvars line
            if line.startswith('.numvars'):
                m = re.search(r'\d+', line)
                if m:
                    num_qubits = int(m.group(0))
                continue
            # gate line
            m = re.match(r'([a-zA-Z_][\w]*)(\([^)]*\))?\s+(.*)', line.strip())
            if not m:
                print(f"Malformed line: {line}"); continue
            token, para, qubits_part = m.groups()
            qubit_strs = [q.strip() for q in qubits_part.split()]
            ## deal with v v+ gate
            if token in {'v', 'v+'}:
                if len(qubit_strs) != 2:
                    print(f"Control/target mismatch in: {line}"); continue
                target = extract_qubit_index(qubit_strs[1])
                ctrls = [qubit_strs[0]]
                gate_enum = gate_str_to_enum[token]
                gate_op = operation_constructor(gate_enum, target, ctrls)
                gate_list.append(gate_op)
                continue
            if token.startswith('t'):
                if len(qubit_strs) == 1:
                    target = extract_qubit_index(qubit_strs[-1])
                    gate_op = operation_constructor(GateType.X, target)
                else:
                    target = extract_qubit_index(qubit_strs[-1])
                    ctrls = [extract_qubit_index(q) for q in qubit_strs[:-1]]
                    gate_op = operation_constructor(GateType.X, target, ctrls)
                gate_list.append(gate_op)
                continue
            if token.startswith('f'): 
                if len(qubit_strs) == 1:
                    print(f"Control/target mismatch in: {line}"); continue
                else:
                    target1 = extract_qubit_index(qubit_strs[-1])
                    target2 = extract_qubit_index(qubit_strs[-2])
                    ctrls = [extract_qubit_index(q) for q in qubit_strs]
                    gate_list.append(
                    operation_constructor(GateType.ARBITRARY, target1, ctrls[-1],
                                          mode='matrix', matrix=[[0, 1], [1, 0]])
                    )
                    gate_list.append(
                    operation_constructor(GateType.ARBITRARY, target2, list(set(ctrls) - set([target2])),
                                          mode='matrix', matrix=[[0, 1], [1, 0]])
                    )
                    gate_list.append(
                    operation_constructor(GateType.ARBITRARY, target1, ctrls[-1],
                                          mode='matrix', matrix=[[0, 1], [1, 0]])
                    )
                continue
            if token.startswith('p'):
                if len(qubit_strs) != 3:
                    print(f"Control/target mismatch in: {line}"); continue
                else:
                    target1 = extract_qubit_index(qubit_strs[-2])
                    target2 = extract_qubit_index(qubit_strs[-1])
                    ctrls1 = [extract_qubit_index(q) for q in qubit_strs[:-2]]
                    ctrls2 = [extract_qubit_index(q) for q in qubit_strs[:-1]]
                    if token == 'pi' :
                        gate_list.append(
                            operation_constructor(GateType.X, target1, ctrls1)
                        )
                        gate_list.append(
                            operation_constructor(GateType.X, target2, ctrls2)
                        )
                        continue
                    else :
                        gate_list.append(
                            operation_constructor(GateType.X, target2, ctrls2)
                        )
                        gate_list.append(
                            operation_constructor(GateType.X, target1, ctrls1)
                        )
                        continue
    if num_qubits is None:
        raise ValueError("No .numvars line found in the file.")
    return num_qubits, gate_list
        
def parse_qasm(path: Path) -> Tuple[int, List[GateOperation]]:
    gate_str_to_enum = {
        'x': GateType.X,  'y': GateType.Y,  'z': GateType.Z,  'h': GateType.H,
        's': GateType.S,  'sdg': GateType.SDG,
        't': GateType.T,  'tdg': GateType.TDG,
        'sx': GateType.SX,
    }

    num_qubits, gate_list = 0, []

    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip().rstrip(';')
            if (not line or line.startswith('//')
                    or line.startswith('OPENQASM')
                    or line.startswith('include')):
                continue

            # qreg
            if line.startswith('qreg'):
                m = re.search(r'qreg\s+\w+\[(\d+)\]', line)
                if m:
                    num_qubits = num_qubits + int(m.group(1))
                continue
            # ignore creg
            if line.startswith('creg'):
                continue

            # ignore barrier
            if line.startswith('barrier'):
                continue
            # ignore measure
            if line.startswith('measure'):
                continue

            m = re.match(r'([a-zA-Z_][\w]*)(\([^)]*\))?\s+(.*)', line)
            if not m:
                print(f"Malformed line: {line}"); continue
            token, param_group, qubits_part = m.groups()
            qubit_strs = [q.strip() for q in qubits_part.split(',')]

            # control
            ctrl_prefix = re.match(r'c+', token)
            num_ctrl = len(ctrl_prefix.group()) if ctrl_prefix else 0
            base_token = token[num_ctrl:]

            #swap -> cnot
            if base_token == 'swap' or base_token == 'cswap':
                if len(qubit_strs) != 2 + num_ctrl:
                    print(f"Control/target mismatch in: {line}"); continue
                ctrls = [extract_qubit_index(q) for q in qubit_strs] 
                target1 = extract_qubit_index(qubit_strs[-2])
                target2 = extract_qubit_index(qubit_strs[-1])
                print(ctrls, target1, target2)
                gate_list.append(
                    operation_constructor(GateType.ARBITRARY, target1, ctrls[-1],
                                          mode='matrix', matrix=[[0, 1], [1, 0]])
                )
                gate_list.append(
                    operation_constructor(GateType.ARBITRARY, target2, list(set(ctrls) - set([target2])),
                                          mode='matrix', matrix=[[0, 1], [1, 0]])
                )
                gate_list.append(
                    operation_constructor(GateType.ARBITRARY, target1, ctrls[-1],
                                          mode='matrix', matrix=[[0, 1], [1, 0]])
                )
                continue

            params = []
            if param_group:                       # "(theta,phi,…)" → list[str]
                params = [p.strip() for p in param_group[1:-1].split(',')]
                params = [eval_expr(p) for p in params]

            if len(qubit_strs) != (num_ctrl + 1):
                print(f"Control/target mismatch in: {line}"); continue
            ctrls = [extract_qubit_index(q) for q in qubit_strs[:-1]]
            target = extract_qubit_index(qubit_strs[-1])

            if not params and base_token in gate_str_to_enum:
                gate_enum = gate_str_to_enum[base_token]
                gate_op = operation_constructor(gate_enum, target,
                                                ctrls if num_ctrl else None)
                gate_list.append(gate_op); continue

            if base_token in {'rx','ry','rz','p','phase','u1','u2','u3'}:
                if base_token == 'rx': M = rot_x(params[0])
                elif base_token == 'ry': M = rot_y(params[0])
                elif base_token == 'rz': M = rot_z(params[0])
                elif base_token in ['p','phase','u1']: M = phase(params[0])
                elif base_token == 'u2':
                    phi, lam = params
                    # U2 gate: RZ(phi) · RY(π/2) · RZ(lam)
                    c, s = math.cos(math.pi/4), math.sin(math.pi/4)
                    M = [[c, -s*cmath.exp(1j*lam)], 
                         [s*cmath.exp(1j*phi), c*cmath.exp(1j*(phi+lam))]]
                elif base_token == 'u3':
                    theta, phi, lam = params
                    # U3 gate: general single-qubit unitary
                    c, s = math.cos(theta/2), math.sin(theta/2)
                    M = [[c, -s*cmath.exp(1j*lam)], 
                         [s*cmath.exp(1j*phi), c*cmath.exp(1j*(phi+lam))]]
                
                gate_list.append(
                    operation_constructor(GateType.ARBITRARY, target,
                                          ctrls if num_ctrl else None,
                                          mode='matrix', matrix=M)
                )
                continue

            if base_token in {'p','rx','ry','rz'} and num_ctrl >= 1:
                if base_token == 'p': M = phase(params[0])
                elif base_token == 'rx': M = rot_x(params[0])
                elif base_token == 'ry': M = rot_y(params[0])
                elif base_token == 'rz': M = rot_z(params[0])
                gate_list.append(
                    operation_constructor(GateType.ARBITRARY, target,
                                          ctrls, mode='matrix', matrix=M)
                )
                continue

            print(f"Skipping unsupported or malformed line: {line}")

    if num_qubits is None:
        raise ValueError("No qreg declaration found in QASM file.")

    return num_qubits, gate_list

def parse_qasm_to_custom(path: Path): ##general parser
    if path.suffix == '.qasm':
        return parse_qasm(path)
    elif path.suffix == '.real':
        return parse_reals(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

