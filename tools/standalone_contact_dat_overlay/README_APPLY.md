
# DAT overlay

Copy these files into your existing `tools/` tree:

```text
standalone_contact/dat_kernels.py
standalone_contact/penetration_free.py
tests/test_dat_geometry.py
tests/test_dat_against_newton.py
tests/test_penetration_free.py
```

Expected existing Group-1 files:

```text
tools/
├── newton/
├── standalone_contact/
│   ├── collision_detector.py
│   └── geometry.py
└── tests/
```

Run from `tools/`:

```bash
pytest -q tests/test_dat_geometry.py
pytest -q tests/test_penetration_free.py
pytest -q tests/test_dat_against_newton.py -s
```

PowerShell GPU run:

```powershell
$env:CONTACT_TEST_DEVICE="cuda:0"
pytest -q tests/test_dat_geometry.py tests/test_penetration_free.py tests/test_dat_against_newton.py -s
```

After Group-1 collision detection:

```python
from standalone_contact.penetration_free import PenetrationFreeTruncator

truncator = PenetrationFreeTruncator(
    particle_count=detector.particle_count,
    edge_count=detector.edge_count,
    collision_info=detector.collision_info,
    device=detector.device,
    parallel_eps=detector.edge_edge_parallel_epsilon,
    conservative_bound_relaxation=0.85,
    contact_margin=contact_margin,
)
```

After solver iterations have accumulated displacement from the collision snapshot:

```python
dx_safe, q_safe, truncation_ts = truncator.truncate(
    snapshot_pos=q_collision_snapshot,
    displacement_in=particle_displacements,
    tri_indices=detector.tri_indices,
    edge_indices=detector.edge_indices,
    enable_self_contact=True,
)
```

If Group-1 auto-resize replaced collision buffer arrays, refresh the struct wrapper:

```python
truncator.set_collision_info(detector.collision_info)
```

For the hot loop, preallocate outputs and use `truncate_into()`.
