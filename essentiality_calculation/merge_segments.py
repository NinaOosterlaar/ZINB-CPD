from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, List
import numpy as np


DEFAULT_STRAINS = ["strain_yEK23", "strain_yEK19", "strain_FD", "strain_dnrp", "strain_ylic137", "strain_yTW001", "strain_yWT03a", "strain_yWT04a"]


def format_threshold(value: float) -> str:
    return f"{value:.2f}"


def read_segments(csv_path: Path) -> List[Dict[str, float]]:
    segments: List[Dict[str, float]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"start_index", "end_index_exclusive", "mu_z_score"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = required.difference(set(reader.fieldnames or []))
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            start = int(row["start_index"])
            end = int(row["end_index_exclusive"])
            length = end - start
            segments.append(
                {
                    "start_index": start,
                    "end_index_exclusive": end,
                    "length": length,
                    "mu_z_score": float(row["mu_z_score"]),
                }
            )

    segments.sort(key=lambda item: item["start_index"])
    return segments


def merge_two_segments(left: Dict[str, float], right: Dict[str, float]) -> Dict[str, float]:
    """Merge two adjacent segments using a length-weighted mean score."""
    total_len = left["length"] + right["length"]
    weighted_mu = (
        left["mu_z_score"] * left["length"] +
        right["mu_z_score"] * right["length"]
    ) / total_len

    return {
        "start_index": left["start_index"],
        "end_index_exclusive": right["end_index_exclusive"],
        "length": total_len,
        "mu_z_score": weighted_mu,
    }

def merge_neighbor_segments(segments, merge_threshold):
    if not segments:
        return []

    merged = [dict(seg) for seg in segments]

    while len(merged) > 1:
        scores = np.array([seg["mu_z_score"] for seg in merged], dtype=float)
        diffs = np.abs(np.diff(scores))

        best_idx = int(np.argmin(diffs))
        best_diff = float(diffs[best_idx])

        if best_diff >= merge_threshold:
            break

        new_segment = merge_two_segments(merged[best_idx], merged[best_idx + 1])
        merged = merged[:best_idx] + [new_segment] + merged[best_idx + 2:]

    return merged


def write_merged_segments(output_path: Path, merged_segments: List[Dict[str, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        fieldnames = ["start_index", "end_index_exclusive", "length", "mu_z_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_segments:
            writer.writerow(
                {
                    "start_index": int(row["start_index"]),
                    "end_index_exclusive": int(row["end_index_exclusive"]),
                    "length": int(row["length"]),
                    "mu_z_score": row["mu_z_score"],
                }
            )


def find_segment_mu_files(strain_dir: Path, input_th: str) -> List[Path]:
    suffix = f"th{input_th}_segment_mu.csv"
    return sorted(
        p for p in strain_dir.rglob("*.csv")
        if p.parent.name == "segment_mu" and p.name.endswith(suffix)
    )


def extract_chromosome(file_path: Path) -> str:
    name = file_path.name
    marker = "_distances_"
    if marker in name:
        return name.split(marker, 1)[0]
    return file_path.parents[3].name


def process_strain(strain_dir: Path, input_th: str, merge_threshold: float) -> int:
    files = find_segment_mu_files(strain_dir, input_th)
    if not files:
        print(f"No segment_mu files found for {strain_dir.name} with th{input_th}")
        return 0

    processed = 0
    merge_th_str = format_threshold(merge_threshold)

    for csv_file in files:
        chromosome = extract_chromosome(csv_file)
        segments = read_segments(csv_file)
        merged = merge_neighbor_segments(segments, merge_threshold)

        out_dir = csv_file.parent.parent / "merged_segments"
        out_file = out_dir / f"{chromosome}_th{input_th}_merged_segments_muZ{merge_th_str}.csv"
        write_merged_segments(out_file, merged)

        processed += 1
        print(
            f"{strain_dir.name} | {chromosome} | in={len(segments)} segments | "
            f"out={len(merged)} segments | {out_file}"
        )

    return processed


def normalize_strain_name(name: str) -> str:
    return name if name.startswith("strain_") else f"strain_{name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge adjacent segments by repeatedly merging the globally most similar pair."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("SATAY_CPD_results/CPD_SATAY_results"),
        help="Base directory containing strain_* folders.",
    )
    parser.add_argument(
        "--strains",
        nargs="+",
        default=DEFAULT_STRAINS,
        help="Strains to process (with or without the 'strain_' prefix).",
    )
    parser.add_argument(
        "--input-th",
        type=float,
        default=3.0,
        help=(
            "Input segmentation threshold from source filenames "
            "(e.g. 10.0 for th10.00 files)."
        ),
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.25,
        help="Merge if the smallest adjacent |delta(mu_z_score)| is smaller than this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir
    input_th = format_threshold(args.input_th)
    strains = [normalize_strain_name(s) for s in args.strains]

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    total_files = 0
    for strain in strains:
        strain_dir = base_dir / strain
        if not strain_dir.exists():
            print(f"Skipping missing strain folder: {strain_dir}")
            continue
        total_files += process_strain(strain_dir, input_th, args.merge_threshold)

    print(f"Done. Processed {total_files} files.")


if __name__ == "__main__":
    main()