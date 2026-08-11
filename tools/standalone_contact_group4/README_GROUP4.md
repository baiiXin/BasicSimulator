
# Group 4: penetration-free stress + benchmark

Add these files to your existing `tools/` tree:

```text
tests/
├── group4_stress_utils.py
└── test_penetration_free_stress.py

benchmarks/
└── benchmark_penetration_free.py
```

Requirements for correctness tests:

```bash
pip install pytest
# IPC Toolkit / ipctk must already be installed in the environment.
```

IPC Toolkit is **test-only**. The standalone runtime remains Warp + NumPy.

## Correctness

CPU:

```bash
pytest -q tests/test_penetration_free_stress.py -s
```

GPU PowerShell:

```powershell
$env:CONTACT_TEST_DEVICE="cuda:0"
pytest -q tests/test_penetration_free_stress.py -s
```

The main test asserts:

```text
initial snapshot:      no self-intersection
solver candidate:      self-intersecting
DAT output q_safe:     no self-intersection
NaN / Inf:             none
collision overflow:    none
```

## Performance

PowerShell:

```powershell
python benchmarks/benchmark_penetration_free.py --device cuda:0
```

Example custom sizes:

```powershell
python benchmarks/benchmark_penetration_free.py `
    --device cuda:0 `
    --n 16 32 48 64 96 `
    --repeats 50
```

The CSV-style output reports:

```text
V, T, E
VT candidate count
EE candidate count
number of truncated vertices
minimum truncation t

BVH refit time
VT detection time
EE detection time
DAT time
full begin+truncate time
```

Important: the timing script explicitly synchronizes the Warp device around each
sample so CUDA's asynchronous launches do not produce meaningless near-zero timings.
