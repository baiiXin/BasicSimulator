
# Full 405-frame DAT processing

## 1. Process garments

Use the margin obtained from the successful single-frame scan:

```bash
python process_garment_sequence.py \
    /path/to/intesection_test \
    /path/to/intesection_test_processed \
    --k 7 \
    --contact-margin 0.049354857 \
    --device cuda:0 \
    --check-every-frame
```

The processor:

```text
safe frame f
    -> raw frame f+1 target
    -> repeat K times:
         collision detection
         residual displacement
         DAT
    -> accept safe frame f+1
```

Human is not loaded into the DAT mesh.

Outputs:

```text
intesection_test_processed/
├── jacket/
│   ├── vertices.npy
│   └── faces.npy
├── tee_sleeveless/
│   ├── vertices.npy
│   └── faces.npy
├── processing_report.csv
└── processing_summary.json
```

## 2. Compare raw vs processed static intersections

```bash
python check_sequence_collisions.py \
    /path/to/intesection_test \
    /path/to/intesection_test_processed
```

It reports, per frame:

```text
11 human-human
12 human-jacket
13 human-tee
22 jacket-jacket
23 jacket-tee
33 tee-tee
```

The human sequence in the processed comparison is always taken from the raw
sequence, because human is intentionally not modified.

Important: the current checker reports a boolean per pair per frame, and the
summary reports the number of frames containing at least one intersection.
It does NOT count the number of intersecting triangle pairs.

IPC Toolkit `CollisionMesh.can_collide` is used for cross-layer groups
(12, 13, 23), so human self-intersections do not contaminate human-garment checks.
