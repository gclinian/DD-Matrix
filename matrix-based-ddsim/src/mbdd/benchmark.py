from pathlib import Path
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Tuple, Union, Iterable, Optional, Sequence
from mbdd.circuit import *

DEFAULT_METHODS = [
    'NumPy + PySparse + Weak Sim',
    'NumPy + PySparse + Strong Sim',
    'CuPy + cuSparse + Weak Sim',
    'CuPy + cuSparse + Strong Sim',
    'CudaQ (StateVec)',
    'CudaQ (TensorNet)'
]

DEFAULT_SUBFIELDS = [
    'Time (s)', 'Execution Time (s)', 'Simulation Time (s)',
    'Avg. CPU Memory Usage (MB)', 'Peak CPU Memory Usage (MB)',
    'Avg. GPU Memory Usage (MB)', 'Peak GPU Memory Usage (MB)',
    'Matrix Size', 'Error'
]

def build_multi_columns(methods=None, subfields=None) -> pd.MultiIndex:
    methods = methods or DEFAULT_METHODS
    subfields = subfields or DEFAULT_SUBFIELDS
    cols = [('Qubits', '')]
    for method in methods:
        for sub in subfields:
            if (method in ['NumPy + PySparse + Weak Sim', 'NumPy + PySparse + Strong Sim']) and (sub in ['Avg. GPU Memory Usage (MB)', 'Peak GPU Memory Usage (MB)']):
                continue
            if (method in ['CudaQ (StateVec)', 'CudaQ (TensorNet)']) and (sub in ['Execution Time (s)', 'Simulation Time (s)', 'Matrix Size']):
                continue
            cols.append((method, sub))
    return pd.MultiIndex.from_tuples(cols)

def execute_qasm_case(
    qasm_path: str | Path,
    flag: Tuple[bool, bool, bool, bool, bool, bool],
    *,
    cudaq_dir: Optional[str | Path] = None,
    methods=DEFAULT_METHODS,
    subfields=DEFAULT_SUBFIELDS,
    verbose: bool = False,
    parser=None,                              # optional single parser
    parser_map: Optional[Dict[str, callable]] = None  # optional {".ext": parse_fn}
) -> Tuple[list, Tuple[bool, bool, bool, bool, bool, bool]]:
    """
    Run one circuit file through the pipeline and return (row, updated_flags).

    If parser_map is provided and contains qasm_path.suffix, that function is used.
    Else if parser is provided, that is used.
    Else falls back to parse_qasm_to_custom.
    """
    methods = methods or DEFAULT_METHODS
    subfields = subfields or DEFAULT_SUBFIELDS
    qasm_path = Path(qasm_path)
    if cudaq_dir is None:
        cudaq_dir = qasm_path.parent
    cudaq_dir = Path(cudaq_dir)
    cudaq_dir_str = str(cudaq_dir if str(cudaq_dir).endswith('/') else str(cudaq_dir) + '/')

    # --- choose parser ---
    parse_fn = None
    if parser_map and qasm_path.suffix in parser_map:
        parse_fn = parser_map[qasm_path.suffix]
    elif parser is not None:
        parse_fn = parser
    else:
        parse_fn = parse_qasm_to_custom  # default

    # Parse
    try:
        NUM_QUBITS, operations = parse_fn(qasm_path)
    except Exception as e:
        row = [
            '--',
            '--', '--', '--', '--', '--', '--', f'parse-error: {e}',
            '--', '--', '--', '--', '--', '--', f'parse-error: {e}',
            '--', '--', '--', '--', '--', '--', '--', '--', f'parse-error: {e}',
            '--', '--', '--', '--', '--', '--', '--', '--', f'parse-error: {e}',
            '--', '--', '--', '--', '--', f'parse-error: {e}',
            '--', '--', '--', '--', '--', f'parse-error: {e}',
        ]
        return row, flag  # leave flags unchanged on parse error

    if verbose:
        print(f"[run] {qasm_path}  | qubits={NUM_QUBITS}")

    INIT_MATRIX_SIZE = 2 * NUM_QUBITS + 1

    if flag[0]:
        np_weak_result = dd_sparse_numpy(operations, INIT_MATRIX_SIZE, NUM_QUBITS, MODE='weak_simulation')
    if flag[1]:
        cp_weak_result = dd_sparse_cupy(operations, INIT_MATRIX_SIZE, NUM_QUBITS, MODE='weak_simulation')
    if flag[2]:
        cudaq_sv_result = cudaq_sv(qasm_path, NUM_QUBITS, MODE='nvidia')
    if flag[3]:
        np_strong_result = dd_sparse_numpy(operations, INIT_MATRIX_SIZE, NUM_QUBITS, MODE='strong_simulation')
    if flag[4]:
        cp_strong_result = dd_sparse_cupy(operations, INIT_MATRIX_SIZE, NUM_QUBITS, MODE='strong_simulation')
    if flag[5]:
        cudaq_tn_result = cudaq_sv(qasm_path, NUM_QUBITS, MODE='tensornet')

    row = [
        NUM_QUBITS,
        np_weak_result['time_spent'] if flag[0] else '--',
        np_weak_result['exec_time']  if flag[0] else '--',
        np_weak_result['sim_time']   if flag[0] else '--',
        np_weak_result['avg_cpu']    if flag[0] else '--',
        np_weak_result['peak_mem']   if flag[0] else '--',
        np_weak_result['matrix_size'] if flag[0] else '--',
        np_weak_result['error']      if flag[0] else '--',
        np_strong_result['time_spent'] if flag[3] else '--',
        np_strong_result['exec_time']  if flag[3] else '--',
        np_strong_result['sim_time']   if flag[3] else '--',
        np_strong_result['avg_cpu']    if flag[3] else '--',
        np_strong_result['peak_mem']   if flag[3] else '--',
        np_strong_result['matrix_size'] if flag[3] else '--',
        np_strong_result['error']      if flag[3] else '--',
        cp_weak_result['time_spent']   if flag[1] else '--',
        cp_weak_result['exec_time']    if flag[1] else '--',
        cp_weak_result['sim_time']     if flag[1] else '--',
        cp_weak_result['avg_cpu']      if flag[1] else '--',
        cp_weak_result['peak_mem']     if flag[1] else '--',
        cp_weak_result['avg_gpu']     if flag[1] else '--',
        cp_weak_result['peak_gpu_mem'] if flag[1] else '--',
        cp_weak_result['matrix_size']  if flag[1] else '--',
        cp_weak_result['error']        if flag[1] else '--',
        cp_strong_result['time_spent']   if flag[4] else '--',
        cp_strong_result['exec_time']    if flag[4] else '--',
        cp_strong_result['sim_time']     if flag[4] else '--',
        cp_strong_result['avg_cpu']      if flag[4] else '--',
        cp_strong_result['peak_mem']     if flag[4] else '--',
        cp_strong_result['avg_gpu']      if flag[4] else '--',
        cp_strong_result['peak_gpu_mem'] if flag[4] else '--',
        cp_strong_result['matrix_size']  if flag[4] else '--',
        cp_strong_result['error']        if flag[4] else '--',
        cudaq_sv_result['time_spent']    if flag[2] else '--',
        cudaq_sv_result['avg_cpu']       if flag[2] else '--',
        cudaq_sv_result['peak_mem']      if flag[2] else '--',
        cudaq_sv_result['avg_gpu']       if flag[2] else '--',
        cudaq_sv_result['peak_gpu_mem']  if flag[2] else '--',
        cudaq_sv_result['error']         if flag[2] else '--',
        cudaq_tn_result['time_spent']    if flag[5] else '--',
        cudaq_tn_result['avg_cpu']       if flag[5] else '--',
        cudaq_tn_result['peak_mem']      if flag[5] else '--',
        cudaq_tn_result['avg_gpu']       if flag[5] else '--',
        cudaq_tn_result['peak_gpu_mem']  if flag[5] else '--',
        cudaq_tn_result['error']         if flag[5] else '--',
    ]

    # Update flags based on errors
    new_flag = (
        flag[0] and (np_weak_result['error'] is None),
        flag[1] and (cp_weak_result['error'] is None),
        flag[2] and (cudaq_sv_result['error'] is None),
        flag[3] and (np_strong_result['error'] is None),
        flag[4] and (cp_strong_result['error'] is None),
        flag[5] and (cudaq_tn_result['error']  is None),
    )
    return row, new_flag

def format_case_index(benchmark_name: str, k: int) -> str:
    i = f"{k:02d}"
    if benchmark_name in ['GHZall', 'GHZzero', 'H2'] and k < 100:
        i = f"{k:03d}"
    return i

def should_stop_early(benchmark_name: str, k: int) -> bool:
    # if benchmark_name == 'Grover' and k == 8:
    #     return True
    return False

def run_autoq_benchmarks(
    benchmarks: dict,
    *,
    manual: bool = False,
    base_path: str = "../ext/Benchmarks/AutoQ",
    methods=DEFAULT_METHODS,
    subfields=DEFAULT_SUBFIELDS,
    max_k: int = 1000,
    verbose: bool = True
) -> Tuple[Dict[str, list], pd.MultiIndex]:
    multi_columns = build_multi_columns(methods, subfields)
    benchmark_data: Dict[str, list] = {}

    for name in benchmarks:
        if manual:
            break
        k = benchmarks[name]
        rows = []
        flag = (True, True, True, True, True, True)  # (npW, cpW, sv, npS, cpS, tn)

        while (flag[0] or flag[1] or flag[3] or flag[4]) and (flag[2] or flag[5]) and k <= max_k:
            if should_stop_early(name, k):
                break
            i = format_case_index(name, k)
            case_dir = f"{base_path}/{name}/{i}/"
            qasm_file = Path(case_dir) / "circuit.qasm"

            row, flag = execute_qasm_case(
                qasm_file,
                flag,
                cudaq_dir=qasm_file,
                methods=methods,
                subfields=subfields,
                verbose=verbose,
            )
            rows.append(row)
            k += 1

        benchmark_data[name] = rows

    return benchmark_data, multi_columns

def run_benchmarks_from_tree(
    benchmark: str,
    tree: Dict[str, int | dict],
    root: str | Path,
    *,
    key_prefix: str = "VQB",
    file_globs: str | Sequence[str] = ("*.qasm", "*.real"),
    methods=DEFAULT_METHODS,
    subfields=DEFAULT_SUBFIELDS,
    verbose: bool = True,
    dry_run: bool = False,
    parser=None,
    parser_map: Optional[Dict[str, callable]] = None,
) -> Tuple[Dict[str, list], pd.MultiIndex]:

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Root not found or not a directory: {root.resolve()}")

    def _iter_leaf_dirs(t: dict, base: Path, parts: Tuple[str, ...] = ()) -> Iterable[Path]:
        for name, sub in t.items():
            here = base.joinpath(*parts, name)
            if sub == 0:
                yield here
            elif isinstance(sub, dict):
                yield from _iter_leaf_dirs(sub, base, parts + (name,))

    multi_columns = build_multi_columns(methods, subfields)
    out: Dict[str, list] = {}
    total_files = 0
    leaf_dirs = list(_iter_leaf_dirs(tree, root))

    patterns = (file_globs,) if isinstance(file_globs, str) else tuple(file_globs)

    for d in leaf_dirs:
        if not d.is_dir():
            continue

        print("d: ", d)

        matches = []
        for pat in patterns:
            matches.extend(d.glob(pat))

        import re

        if benchmark == 'MQT':

            def qubit_key(p):
                m = re.search(r'_(\d+)(?=\.qasm$)', p.name)
                n = int(m.group(1)) if m else float('inf') 
                return (n, p.name.lower())

            files = sorted({p.resolve() for p in matches}, key=qubit_key)
            
        elif benchmark == 'RevLib':

            def qubit_key(p):
                s = p.name
                m = re.search(r'v\d+_(\d+)_renamed\.real$', s, flags=re.IGNORECASE)
                n = int(m.group(1)) if m else float('inf')
                return (n, s.lower())

            files = sorted({p.resolve() for p in matches}, key=qubit_key)

        elif benchmark == 'VeriQ':

            if d.stem in ['adder']:
                def qubit_key(p):
                    m = re.search(r'_n(\d+)$', p.stem)
                    n = int(m.group(1)) if m else float('inf')
                    return (n, p.name.lower())

            elif d.stem in ['qv']:
                def qubit_key(p):
                    m = re.search(r'_n(\d+)(?:_|$)', p.stem)
                    n = int(m.group(1)) if m else float('inf')
                    return (n, p.name.lower())

            elif d.stem == 'rand_cliff':
                def qubit_key(p):
                    m = re.search(r'_(\d+)(?=_)', p.stem)
                    n = int(m.group(1)) if m else float('inf')
                    return (n, p.name.lower())

            else:
                def qubit_key(p):
                    m = re.search(r'_(\d+)$', p.stem)
                    n = int(m.group(1)) if m else float('inf')
                    return (n, p.name.lower())

            files = sorted({p.resolve() for p in matches}, key=qubit_key)

        print(files)

        # files = sorted({p.resolve() for p in matches}, key=lambda p: p.name.lower())
        if not files:
            continue

        if verbose:
            rel = d.relative_to(root).as_posix()
            print(f"[dir] {rel} ({len(files)} file(s))")

        rows = []
        flag = (True, True, True, True, True, True)
        for fpath in files:
            if (flag[0] or flag[1] or flag[3] or flag[4]) == 0 or (flag[2] or flag[5]) == 0:
                break
            total_files += 1
            if dry_run:
                rows.append(['--'] + ['--'] * (len(multi_columns) - 1))
                continue

            row, flag = execute_qasm_case(
                fpath,
                flag,
                cudaq_dir=str(d)+"/"+str(fpath.stem),
                methods=methods,
                subfields=subfields,
                verbose=verbose,
                parser=parser,
                parser_map=parser_map,
            )
            rows.append(row)

        key = f"{key_prefix}:{d.relative_to(root).as_posix()}"
        out[key] = rows

    if verbose:
        print(f"[done] VeriQBench processed {total_files} file(s) in {len(leaf_dirs)} leaf dir(s).")
    return out, multi_columns

def save_benchmark_dict_to_excel(
    data: Dict[str, list],
    columns: pd.MultiIndex | list,
    out_path: str | Path,
    *,
    index: bool = True,
    add_summary: bool = True,
):
    """
    Writes a dict-of-rows to an .xlsx where each key becomes a sheet.
    - data: {key -> rows(list)}
    - columns: your MultiIndex from build_multi_columns / the runners
    - out_path: output .xlsx path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Excel sheet name rules: max 31 chars, forbid [: \ / ? * [ ]]
    invalid = re.compile(r'[:\\/*?\[\]]')
    used = set()

    def make_sheet_name(key: str) -> str:
        name = invalid.sub('·', str(key)).strip() or "Sheet"
        # keep the rightmost part if too long (often most specific)
        if len(name) > 31:
            name = name[-31:]
        base = name
        # ensure unique across workbook
        n = 1
        while name in used:
            tail = f"~{n}"
            name = base[:31-len(tail)] + tail
            n += 1
        used.add(name)
        return name

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if add_summary:
            pd.DataFrame(
                [{"key": k, "#rows": (len(v) if v is not None else 0)} for k, v in data.items()]
            ).to_excel(writer, sheet_name="__index__", index=True)

        for key, rows in data.items():
            # Build DF with your provided columns; fall back to plain DF if misaligned
            try:
                df = pd.DataFrame(rows, columns=columns)
            except Exception:
                df = pd.DataFrame(rows)
            if df.empty and isinstance(columns, pd.MultiIndex):
                # write headers even for empty sets
                df = pd.DataFrame(columns=columns)
            df.to_excel(writer, sheet_name=make_sheet_name(key), index=index)

    print(f"Wrote {len(data)} sheet(s) → {out_path.resolve()}")

def build_tree_or_zero(dir_path: Path):
    """
    Return a nested dict of subfolders for dir_path.
    If a folder has no (non-symlink) subfolders, return 0 instead of {}.
    """
    children = {}
    for p in sorted(dir_path.iterdir(), key=lambda x: x.name.lower()):
        try:
            if p.is_symlink() or not p.is_dir():
                continue
        except OSError:
            continue
        children[p.name] = build_tree_or_zero(p)
    return 0 if not children else children
