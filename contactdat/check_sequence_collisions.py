
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import ipctk


GROUPS = ("11", "12", "13", "22", "23", "33")
NAMES = {1: "human", 2: "jacket", 3: "tee_sleeveless"}


def unique_edges(faces):
    faces = np.asarray(faces, dtype=np.int32)
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    e = np.sort(e, axis=1)
    return np.asfortranarray(np.unique(e, axis=0), dtype=np.int32)


class PairOracle:
    def __init__(self, rest_a, faces_a, owner_a, rest_b=None, faces_b=None, owner_b=None):
        if rest_b is None:
            self.cross = False
            self.na = len(rest_a)
            self.rest = np.asfortranarray(rest_a, dtype=np.float64)
            self.faces = np.asfortranarray(faces_a, dtype=np.int32)
        else:
            self.cross = True
            self.na = len(rest_a)
            self.rest = np.asfortranarray(
                np.vstack([rest_a, rest_b]), dtype=np.float64
            )
            self.faces = np.asfortranarray(
                np.vstack([faces_a, faces_b + len(rest_a)]), dtype=np.int32
            )

        self.edges = unique_edges(self.faces)
        self.mesh = ipctk.CollisionMesh(self.rest, self.edges, self.faces)

        if self.cross:
            na = self.na
            # Only cross-owner primitive pairs are allowed to collide.
            self.mesh.can_collide = lambda i, j, na=na: (int(i) < na) != (int(j) < na)

    def has_intersections(self, va, vb=None):
        if vb is None:
            V = np.asfortranarray(va, dtype=np.float64)
        else:
            V = np.asfortranarray(np.vstack([va, vb]), dtype=np.float64)
        return bool(ipctk.has_intersections(self.mesh, V))


def load_topology(root: Path, name: str):
    faces = np.asarray(np.load(root / name / "faces.npy"), dtype=np.int32)
    seq = np.load(root / name / "vertices.npy", mmap_mode="r")
    return seq, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("processed_root", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--groups",
        nargs="+",
        default=list(GROUPS),
        choices=list(GROUPS),
    )
    args = ap.parse_args()

    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else processed_root / "collision_comparison"
    )
    output.mkdir(parents=True, exist_ok=True)

    raw_seq = {}
    raw_faces = {}
    proc_seq = {}
    proc_faces = {}

    for owner, name in NAMES.items():
        raw_seq[owner], raw_faces[owner] = load_topology(raw_root, name)

        if owner == 1:
            # Human is intentionally not processed.
            proc_seq[owner] = raw_seq[owner]
            proc_faces[owner] = raw_faces[owner]
        else:
            proc_seq[owner], proc_faces[owner] = load_topology(processed_root, name)

    frame_count = int(raw_seq[1].shape[0])
    for owner in (1, 2, 3):
        if raw_seq[owner].shape[0] != frame_count or proc_seq[owner].shape[0] != frame_count:
            raise ValueError("Frame counts do not match.")

    # Build one exact static-intersection oracle per requested layer pair.
    raw_oracles = {}
    proc_oracles = {}

    for g in args.groups:
        a, b = int(g[0]), int(g[1])

        ra = np.asarray(raw_seq[a][0], dtype=np.float64)
        pa = np.asarray(proc_seq[a][0], dtype=np.float64)

        if a == b:
            raw_oracles[g] = PairOracle(ra, raw_faces[a], a)
            proc_oracles[g] = PairOracle(pa, proc_faces[a], a)
        else:
            rb = np.asarray(raw_seq[b][0], dtype=np.float64)
            pb = np.asarray(proc_seq[b][0], dtype=np.float64)

            raw_oracles[g] = PairOracle(
                ra, raw_faces[a], a,
                rb, raw_faces[b], b,
            )
            proc_oracles[g] = PairOracle(
                pa, proc_faces[a], a,
                pb, proc_faces[b], b,
            )

    rows = []
    raw_frame_counts = {g: 0 for g in args.groups}
    proc_frame_counts = {g: 0 for g in args.groups}

    for frame in range(frame_count):
        row = {"frame": frame}

        for g in args.groups:
            a, b = int(g[0]), int(g[1])

            va_raw = np.asarray(raw_seq[a][frame], dtype=np.float64)
            va_proc = np.asarray(proc_seq[a][frame], dtype=np.float64)

            if a == b:
                raw_hit = raw_oracles[g].has_intersections(va_raw)
                proc_hit = proc_oracles[g].has_intersections(va_proc)
            else:
                vb_raw = np.asarray(raw_seq[b][frame], dtype=np.float64)
                vb_proc = np.asarray(proc_seq[b][frame], dtype=np.float64)

                raw_hit = raw_oracles[g].has_intersections(va_raw, vb_raw)
                proc_hit = proc_oracles[g].has_intersections(va_proc, vb_proc)

            row[f"raw_{g}"] = raw_hit
            row[f"processed_{g}"] = proc_hit
            raw_frame_counts[g] += int(raw_hit)
            proc_frame_counts[g] += int(proc_hit)

        rows.append(row)

        if frame % 10 == 0 or frame == frame_count - 1:
            compact = " ".join(
                f"{g}:{int(row[f'raw_{g}'])}->{int(row[f'processed_{g}'])}"
                for g in args.groups
            )
            print(f"[{frame:4d}/{frame_count-1}] {compact}")

    csv_path = output / "collision_by_frame.csv"
    fieldnames = ["frame"]
    for g in args.groups:
        fieldnames.extend([f"raw_{g}", f"processed_{g}"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "frame_count": frame_count,
        "groups": list(args.groups),
        "meaning": (
            "Per-frame boolean static intersections. "
            "intersecting_frame_count is number of frames with >=1 intersection, "
            "not the number of primitive intersection pairs."
        ),
        "raw_intersecting_frame_count": raw_frame_counts,
        "processed_intersecting_frame_count": proc_frame_counts,
    }
    json_path = output / "collision_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Intersecting frame counts ===")
    for g in args.groups:
        print(
            f"{g}: raw={raw_frame_counts[g]:4d} "
            f"processed={proc_frame_counts[g]:4d}"
        )
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
