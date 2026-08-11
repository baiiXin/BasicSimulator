
# Group 3 integration overlay

Add:

```text
standalone_contact/penetration_free_contact.py
tests/test_penetration_free_contact.py
```

This assumes Group 1 + Group 2 already exist:

```text
standalone_contact/
├── collision_detector.py
├── geometry.py
├── dat_kernels.py
├── penetration_free.py
└── penetration_free_contact.py
```

Core API:

```python
contact.begin(q_current)

dx = solver(...)  # dx must be relative to contact.q_snapshot
dx_safe, q_safe, truncation_ts = contact.truncate(dx)

# Explicitly start a new collision epoch when desired:
contact.redetect(q_safe)
```

CPU test:

```bash
pytest -q tests/test_penetration_free_contact.py
```

GPU / PowerShell:

```powershell
$env:CONTACT_TEST_DEVICE="cuda:0"
pytest -q tests/test_penetration_free_contact.py -s
```

Real Group-1 detector construction:

```python
contact = PenetrationFreeContact(
    detector,
    particle_count=particle_count,
    edge_count=edge_count,
    device=device,
    contact_margin=contact_margin,
    conservative_bound_relaxation=0.85,
    rest_q=q_rest,
    rest_exclusion_radius=rest_exclusion_radius,
)
```

Expected detector interface:

```python
detector.refit(q)

detector.vertex_triangle_collision_detection(
    query_radius,
    min_query_radius=...,
    min_distance_filtering_ref_pos=...,
)

detector.edge_edge_collision_detection(
    query_radius,
    min_query_radius=...,
    min_distance_filtering_ref_pos=...,
)
```

If your Group-1 standalone detector renamed these methods, only `begin()` needs
a thin adapter; the snapshot/DAT state model is unchanged.
