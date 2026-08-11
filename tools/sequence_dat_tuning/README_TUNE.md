
# Single-frame DAT K tuning

Run from your `tools/` directory:

```bash
python tune_sequence_truncation.py /path/to/00396 \
    --frame 1 \
    --k-max 20 \
    --device cuda:0
```

This means raw frame 0 -> raw frame 1.

Only the listed garments are processed; human is ignored:

```bash
--garments jacket tee_sleeveless
```

If `--contact-margin` is omitted:

```text
contact_margin = 0.5 * median(initial combined-cloth edge length)
```

Override explicitly if desired:

```bash
python tune_sequence_truncation.py /path/to/00396 \
    --frame 1 \
    --k-max 20 \
    --device cuda:0 \
    --contact-margin 0.01
```

Each K is exactly:

```text
1 collision detection + 1 DAT truncation
```

The residual is always:

```text
raw target frame - latest accepted safe position
```

Default outputs go to:

```text
00396/dat_tuning/
```

Files:

```text
frame_0000_to_0001_k_scan.csv
frame_0000_to_0001_k_scan.json
frame_0000_to_0001_residual.png
frame_0000_to_0001_accepted_positions.npy
```

The script reports per K:

```text
remaining displacement ratio
max per-vertex residual
truncated vertex count
min truncation t
VT / EE candidates
collision detection time
DAT time
IPC static intersection flag
buffer overflow, if exposed by the detector
```

Important: source data are float64, but the current standalone Warp kernels use
`wp.vec3` (float32 runtime). Accepted positions are saved back as float64.
