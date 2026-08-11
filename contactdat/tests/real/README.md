# Real-data regressions

These regressions cover selected D-LAYERS garment transitions that are not
represented by the synthetic unit tests.

- `frame0_residual_vs_truncation.py` records the relationship between the DAT
  truncation count and the remaining displacement for raw frame 0 to frame 1.
- `frame63_known_failure.py` is reserved for the validated behavior in which
  micro-steps 1 through 14 are non-intersecting and micro-step 15 intersects.

The scripts require a user-supplied D-LAYERS sample. Place local-only data
under `tests/real/data/` or provide its location through the script arguments.
The three `fail_q_*.npy` files retained here are the minimal saved state used
by the validated frame-63 micro-step replay.

Do not redistribute D-LAYERS-derived arrays publicly until the dataset license
has been checked. If redistribution is not permitted, remove the saved arrays
and require users to prepare them locally.

The canonical real-regression scripts and their import paths still need the
packaging migration described in `docs/repository-layout.md`; that code change
is intentionally outside this directory-only reorganization.
