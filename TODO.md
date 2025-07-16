Several things to do:
- Remove mapping:
    - use selection matrix with L and R
    - the size is (NUM_QUBITS+1) * MATRIX_SIZE
- Combine CuPy with PySparse to compare the performance with CudaQ and to test the upper bound of both models
- Try CudaQ tensor network-based simulator
- Increase sampling size to check if state vector-based simulator actually performs better on larger input (based on previous experiment)
- Explain Masking in a clear manner
- ...