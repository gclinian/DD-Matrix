## A Weighted Binary Decision Diagram-Based Quantum Simulator

- To start the environment, run `uv sync` and make sure all the dependencies are installed.
- To install the simulator package, run `pip install -e matrix-based-ddsim`.
- To manually test the circuit, go to `/src/run.ipynb` and run.
- To run the benchmark set:
    1. Include the external benchmark set as a submodule by running `git submodule --init --recursive`
    2. Check the existence of the benchmark set in `/ext/Benchmarks/`
    3. Go to `/src/bench.ipynb` and run.
