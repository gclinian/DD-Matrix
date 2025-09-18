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

def cudaq_result_to_probs(cudaq_result: str, sample_size: int) -> Dict[str, float]:
    """
    Convert cudaQ result string to a dictionary of probabilities.
    """
    cudaq_result = cudaq_result.strip().lstrip("{").rstrip("}").strip()
    cudaq_probs = {}
    if cudaq_result:
        for pair in cudaq_result.split():
            if ":" not in pair:
                continue
            state, prob_str = pair.split(":")
            cudaq_probs[state] = float(prob_str) / sample_size
    return cudaq_probs
    
def dd_result_to_probs(dd_result: Dict[Tuple[int, ...], complex]) -> Dict[str, float]:
    """
    Convert dd_sparse result dictionary to a dictionary of probabilities.
    """
    return {
        "".join(map(str, state)): abs(amp) ** 2
        for state, amp in dd_result.items()
    }

def dd_and_cudaq_is_equal(
    my_result: Dict[Tuple[int, ...], complex],
    cudaq_result: str,
    sample_size: int = 1024,
    tol: float = 1e-6
) -> bool:
    """
    Compare probabilities from a DD-sparse simulator and a cudaQ simulator.

    Parameters
    ----------
    my_result : dict
        Key: tuple of 0/1 ints representing a basis state.
        Value: complex amplitude from dd_sparse.
    cudaq_result : str
        cudaQ output string, e.g. "{ 11:500 00:500 }".
        Each probability is scaled by 1000.
    tol : float, optional
        Allowed absolute difference between probabilities.

    Returns
    -------
    bool
        True if every state's probabilities match within `tol`, else False.
    """

    cudaq_probs = cudaq_result_to_probs(cudaq_result, sample_size)
    dd_probs = dd_result_to_probs(my_result)
    
    all_states = set(dd_probs) | set(cudaq_probs)

    for st in all_states:
        p_dd = dd_probs.get(st, 0.0)
        p_cq = cudaq_probs.get(st, 0.0)
        if abs(p_dd - p_cq) > tol:
            return False

    return True

def dd_sample_result_initializer(NUM_QUBITS: int) -> Dict[str, int]:
    """
    Initialize the sample result dictionary with all possible states.
    """
    dd_sample_result = {}
    
    for i in range(2 ** NUM_QUBITS):
        state = bin(i)[2:].zfill(NUM_QUBITS)
        dd_sample_result[state] = 0
    
    return dd_sample_result

def dd_sampler(dd_result: Dict[Tuple[int, ...], complex], sample_size: int) -> Dict[str, float]:
    """
    Sample from the DD result dictionary to get a sample distribution.
    Parameters
    ----------
    dd_result : dict
        Key: tuple of 0/1 ints representing a basis state.
        Value: complex amplitude from dd_sparse.
    sample_size : int
        Number of samples to draw from the distribution.
    Returns
    -------
    dict
        Key: tuple of 0/1 ints representing a basis state.
        Value: float representing the probability of that state in the sample.
    """
    
    dd_probs = dd_result_to_probs(dd_result)
    dd_sample_result = dd_sample_result_initializer()
    
    sampled_states = random.choices(list(dd_probs.keys()), weights = list(dd_probs.values()), k = sample_size)

    for state in sampled_states:
        dd_sample_result[state] += 1
    
    for state in dd_sample_result:
        dd_sample_result[state] /= sample_size
    
    return dd_sample_result

def compare_strong_weak(strong_amp: dict, weak_prob: dict):
    
    tuple_to_str = lambda t: "".join(map(str, t))
    strong_prob = {tuple_to_str(k): float(abs(v)**2) for k, v in strong_amp.items()}
    keys = sorted(set(strong_prob) | set(weak_prob))
    rows = []
    for k in keys:
        ps = strong_prob.get(k, 0.0)
        pw = weak_prob.get(k, 0.0)
        rel = abs(ps - pw) / ps if ps > 0 else np.nan
        rows.append((k, ps, pw, rel))
    # summary
    L1 = sum(abs(strong_prob.get(k,0)-weak_prob.get(k,0)) for k in keys)
    KL = sum(
        strong_prob[k]*np.log(strong_prob[k]/weak_prob[k])
        for k in keys if strong_prob.get(k,0)>0 and weak_prob.get(k,0)>0
    )
    return rows, {"L1": L1, "KL": KL, "sum_strong": sum(strong_prob.values()), "sum_weak": sum(weak_prob.values())}