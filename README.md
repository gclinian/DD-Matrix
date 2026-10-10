# DD-Matrix

**A quantum circuit simulator that stores a weighted binary decision diagram (DD) as a pair of sparse matrices, so gate application and simulation become sparse linear algebra that runs on the CPU (NumPy/SciPy) or the GPU (CuPy).**

Decision-diagram simulators (e.g. DDSIM) compress quantum states well, but they are pointer-chasing graph algorithms that are hard to parallelize. DD-Matrix keeps the DD representation and replaces the graph traversal with sparse matrix operations:

| | |
|---|---|
| **State** | A weighted DD encoded as two sparse adjacency matrices: `L` (0-edges) and `R` (1-edges), plus a `Mapping` that says which nodes sit on which qubit level. |
| **Gates** | A single-qubit gate `U = [[a, b], [c, d]]` on qubit *t* rewrites the outgoing edges of every node on level *t*: `L' = L·A + R·B`, `R' = L·C + R·D`, where `A…D` are diagonal matrices holding the gate entries on those nodes. Controlled gates additionally split nodes, so the matrices grow as needed. |
| **Strong simulation** | Computes the full amplitude vector by walking root-to-terminal paths with sparse matrix–vector products (with precomputed two-step products such as `L·L`, `L·R`). |
| **Weak simulation** | Samples measurement outcomes (1024 shots by default) by propagating path weights through the DD, without enumerating every amplitude. |
| **Backends** | The same code runs with `numpy` + `scipy.sparse` or `cupy` + `cupyx.scipy.sparse`. |

Results are benchmarked against NVIDIA CUDA-Q's state-vector and tensor-network simulators.

## Repository layout

```
DD-Matrix/
├── matrix-based-ddsim/        # the simulator, installable as the `mbdd` package
│   ├── src/mbdd/
│   │   ├── gates.py           # gate types, GateOperation constructors, DD initialization
│   │   ├── circuit.py         # dd_sparse_numpy / dd_sparse_cupy (main entry points), CUDA-Q baseline
│   │   ├── simulator.py       # strong (all-path) and weak (sampling) simulation
│   │   ├── parser.py          # OpenQASM 2.0 (.qasm) and RevLib (.real) parsers
│   │   ├── comparator.py      # compare results against CUDA-Q / between strong and weak modes
│   │   └── benchmark.py       # batch benchmark runner, Excel export
│   └── tests/
├── src/
│   ├── run.ipynb              # run a hand-written circuit
│   └── bench.ipynb            # run the benchmark suites
├── references/                # CuPy sparse experiments
└── ext/Benchmarks/            # benchmark circuits (git submodule)
```

## Installation

Requires Python ≥ 3.12. The GPU backend and the CUDA-Q baselines need an NVIDIA GPU with CUDA 12.

```bash
git clone https://github.com/gclinian/DD-Matrix.git
cd DD-Matrix

uv sync                                 # environment from pyproject.toml / uv.lock
uv pip install -e matrix-based-ddsim    # the mbdd package
uv pip install psutil nvidia-ml-py      # used for CPU/GPU usage monitoring; not yet in pyproject.toml
```

CPU-only machines can skip `uv sync` (the CUDA packages won't install) and just run:

```bash
pip install numpy scipy pandas openpyxl psutil nvidia-ml-py
pip install -e matrix-based-ddsim
```

## Quick start

Build a 3-qubit GHZ circuit and simulate it both ways:

```python
from mbdd.gates import GateType, operation_constructor
from mbdd.circuit import dd_sparse_numpy   # or dd_sparse_cupy for the GPU

n = 3
ops = [
    operation_constructor(GateType.H, 0),        # H on qubit 0
    operation_constructor(GateType.X, 1, [0]),   # CNOT, control 0 → target 1
    operation_constructor(GateType.X, 2, [1]),   # CNOT, control 1 → target 2
]

strong = dd_sparse_numpy(ops, 2 * n + 1, n, MODE="strong_simulation")
print(strong["sim_result"])
# {(1, 1, 1): (0.7071+0j), (0, 0, 0): (0.7071+0j)}

weak = dd_sparse_numpy(ops, 2 * n + 1, n, MODE="weak_simulation")
print(weak["sim_result"])
# {'111': 522, '000': 502}   (1024 shots)
```

The second argument is the initial matrix size, `2 * NUM_QUBITS + 1`. Besides `sim_result`, the returned dict has timing (`time_spent`, `exec_time`, `sim_time`), resource usage (`avg_cpu`, `peak_mem`, plus `avg_gpu` / `peak_gpu_mem` on CuPy), the final `matrix_size`, and `error`. Each run times out after 60 s.

To simulate a circuit file instead:

```python
from pathlib import Path
from mbdd.parser import parse_qasm_to_custom

n, ops = parse_qasm_to_custom(Path("circuit.qasm"))   # .qasm or .real
result = dd_sparse_numpy(ops, 2 * n + 1, n, MODE="strong_simulation")
```

### Supported gates

- Single-qubit: `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `rx`, `ry`, `rz`, `p`/`u1`, `u2`, `u3` (plus `v`, `v+` from RevLib)
- Controlled: any of the above with one or more controls via the `c` prefix (`cx`, `ccx`, `cccx`, `crz`, `cp`, …)
- `swap`, `cswap`

## Benchmarks

The benchmark circuits are in a separate repository, pulled in as a submodule:

```bash
git submodule update --init --recursive
```

Then open `src/bench.ipynb`. It runs suites from AutoQ (BV, GHZ, Grover, MCToffoli, …), MQT Bench (DJ, QAOA, QFT, QNN, Shor, VQE), VeriQBench and RevLib, and writes one Excel sheet per suite to `result/`. Each circuit is run through six methods:

- DD-Matrix, NumPy, weak / strong simulation
- DD-Matrix, CuPy, weak / strong simulation
- CUDA-Q state vector (`nvidia` target)
- CUDA-Q tensor network (`tensornet` target)

## Tests

```bash
cd matrix-based-ddsim
pytest
```

## Contributors

[@maxyhwu](https://github.com/maxyhwu) · [@gclinian](https://github.com/gclinian) · [@daniel991506](https://github.com/daniel991506)
