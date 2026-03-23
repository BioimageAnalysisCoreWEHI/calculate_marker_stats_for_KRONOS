"""
compute_marker_stats.py
-----------------------
Compute per-marker mean and std (in 0-1 range) from COMET OME-TIFF images
for use in KRONOS marker_metadata.csv.

Normalization strategy (per KRONOS GitHub issue):
  1. Divide raw pixel values by the dtype maximum (e.g. 65535.0 for uint16)
     to scale to [0, 1]. This value should match "marker_max_values" in your
     KRONOS config.
  2. Pool all normalized pixels across ALL images and samples per marker.
  3. Compute global mean and std on the pooled normalized pixels.

At inference, KRONOS applies: (img / marker_max_value - mean) / std

Usage:
    python compute_marker_stats.py \
        --image_dir /path/to/ome_tiffs \
        --output marker_stats.csv \
        --pattern "*.ome.tiff" \
        --dtype_max 65535.0

    # Merge with existing KRONOS metadata to find new markers:
    python compute_marker_stats.py \
        --image_dir /path/to/ome_tiffs \
        --output marker_stats.csv \
        --existing_metadata marker_metadata.csv
"""

import argparse
import multiprocessing as mp
import queue
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


# ---------------------------------------------------------------------------
# OME-TIFF helpers
# ---------------------------------------------------------------------------

DTYPE_MAX = {
    "uint8":   255.0,
    "uint16":  65535.0,
    "uint32":  4294967295.0,
    "float32": 1.0,
    "float64": 1.0,
}


def parse_marker_names_from_ome(tif: tifffile.TiffFile) -> list:
    """
    Extract ordered channel names from the OME-XML metadata embedded in a
    COMET OME-TIFF. Returns a list aligned with the channel axis of the array.
    Falls back to an empty list if parsing fails.
    """
    try:
        ome_xml = tif.ome_metadata
        root = ET.fromstring(ome_xml)

        # OME namespace varies by version — find it dynamically
        ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
        ns_prefix = f"{{{ns}}}" if ns else ""

        image    = root.find(f".//{ns_prefix}Image")
        pixels   = image.find(f"{ns_prefix}Pixels")
        channels = pixels.findall(f"{ns_prefix}Channel")

        names = []
        for ch in channels:
            # COMET typically stores marker name in "Name" or "Fluor"
            name = ch.get("Name") or ch.get("Fluor") or ch.get("ID", "")
            names.append(name.upper().strip())
        return names

    except Exception as e:
        print(f"  [warn] Could not parse OME-XML channel names: {e}")
        return []


def read_ome_tiff(path: Path):
    """
    Read a COMET OME-TIFF.
    Returns:
        image     : array of shape (C, H, W), original dtype preserved
        markers   : list of marker/channel names, length C
        dtype_max : maximum value for this dtype (e.g. 65535.0 for uint16)
    """
    with tifffile.TiffFile(path) as tif:
        marker_names = parse_marker_names_from_ome(tif)
        image = tif.asarray()
        raw_dtype = str(image.dtype)

    # Determine dtype max before any casting
    max_val = DTYPE_MAX.get(raw_dtype)
    if max_val is None:
        print(f"  [warn] Unrecognised dtype '{raw_dtype}', defaulting to 65535.0")
        max_val = 65535.0

    # Normalise dimension ordering to (C, H, W)
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    elif image.ndim == 3:
        pass
    elif image.ndim == 4:
        image = image[0]        # (Z or T, C, H, W) — take first slice
    elif image.ndim == 5:
        image = image[0, 0]     # (T, Z, C, H, W)

    n_channels = image.shape[0]

    if len(marker_names) != n_channels:
        print(f"  [warn] Channel name count ({len(marker_names)}) != "
              f"image channels ({n_channels}). Using generic names.")
        marker_names = [f"CH_{i:03d}" for i in range(n_channels)]

    return image, marker_names, max_val


# ---------------------------------------------------------------------------
# Stats accumulation
# ---------------------------------------------------------------------------

def _compute_file_stats(path_str: str, dtype_max_override=None):
    """
    Compute marker-level first and second moments for a single OME-TIFF.
    Returns:
        marker_stats: {marker_name -> (sum, sumsq, count)}
        max_val: dtype max used for normalization
    """
    path = Path(path_str)
    image, marker_names, file_max = read_ome_tiff(path)
    max_val = dtype_max_override if dtype_max_override is not None else file_max

    marker_stats = {}
    for ch_idx, marker in enumerate(marker_names):
        channel = image[ch_idx].astype(np.float64, copy=False) / max_val
        channel_sum = float(np.sum(channel, dtype=np.float64))
        channel_sumsq = float(np.sum(np.multiply(channel, channel, dtype=np.float64), dtype=np.float64))
        channel_count = int(channel.size)
        marker_stats[marker] = (channel_sum, channel_sumsq, channel_count)

    del image
    return marker_stats, max_val


def _worker_compute_file_stats(path_str: str, dtype_max_override, queue):
    """
    Worker entrypoint for subprocess-isolated image parsing.
    If low-level TIFF decode crashes (e.g. SIGBUS), the parent survives and
    can skip that file.
    """
    try:
        marker_stats, max_val = _compute_file_stats(path_str, dtype_max_override)
        queue.put({
            "ok": True,
            "path": path_str,
            "marker_stats": marker_stats,
            "max_val": max_val,
            "error": None,
        })
    except Exception as e:
        queue.put({
            "ok": False,
            "path": path_str,
            "marker_stats": None,
            "max_val": None,
            "error": str(e),
        })


def accumulate_moments(image_paths: list, dtype_max_override=None):
    """
    Iterate over OME-TIFFs and accumulate per-marker moments:
    sum(x), sum(x^2), n on normalized pixels x in [0, 1].

    Each file is processed in an isolated subprocess so a hard crash from a
    single bad TIFF does not terminate the whole run.

    Returns:
        moment_store : { marker_name -> {sum, sumsq, count} }
        dtype_max   : the dtype max value used (for reporting / KRONOS config)
        processed_count : number of files successfully processed
        skipped_count   : number of files skipped due read/parse failure
    """
    moment_store = {}
    detected_max = None
    processed_count = 0
    skipped_count = 0
    ctx = mp.get_context("spawn")

    for path in tqdm(image_paths, desc="Reading images"):
        queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_worker_compute_file_stats,
            args=(str(path), dtype_max_override, queue),
        )
        proc.start()
        proc.join()

        if proc.exitcode != 0:
            print(f"  [error] Skipping {path.name}: reader subprocess exited with code {proc.exitcode}")
            skipped_count += 1
            if proc.is_alive():
                proc.terminate()
            queue.close()
            continue

        try:
            result = queue.get(timeout=5)
        except queue.Empty:
            print(f"  [error] Skipping {path.name}: reader returned no result")
            skipped_count += 1
            queue.close()
            queue.join_thread()
            continue

        queue.close()
        queue.join_thread()

        if not result["ok"]:
            print(f"  [error] Skipping {path.name}: {result['error']}")
            skipped_count += 1
            continue

        marker_stats = result["marker_stats"]
        max_val = result["max_val"]
        processed_count += 1

        if detected_max is None:
            detected_max = max_val
        elif detected_max != max_val and dtype_max_override is None:
            print(f"  [warn] {path.name} dtype max {max_val} differs from "
                  f"first image ({detected_max}). Use --dtype_max to fix.")

        for marker, (channel_sum, channel_sumsq, channel_count) in marker_stats.items():
            if marker not in moment_store:
                moment_store[marker] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
            moment_store[marker]["sum"] += channel_sum
            moment_store[marker]["sumsq"] += channel_sumsq
            moment_store[marker]["count"] += channel_count

    final_max = dtype_max_override if dtype_max_override is not None else (detected_max or 65535.0)
    return moment_store, final_max, processed_count, skipped_count


def compute_stats(moment_store: dict) -> pd.DataFrame:
    """
    Compute mean/std from per-marker accumulated moments.
    """
    rows = []
    for marker in sorted(moment_store.keys()):
        total_sum = float(moment_store[marker]["sum"])
        total_sumsq = float(moment_store[marker]["sumsq"])
        total_count = int(moment_store[marker]["count"])

        if total_count == 0:
            continue

        mean = total_sum / total_count
        variance = max((total_sumsq / total_count) - (mean * mean), 0.0)
        std = float(np.sqrt(variance))

        rows.append({
            "marker_name": marker,
            "marker_mean": round(float(mean), 6),
            "marker_std":  round(float(std),  6),
            "n_pixels":    total_count,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute KRONOS marker_mean / marker_std from COMET OME-TIFFs"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing OME-TIFF files (searched recursively)"
    )
    parser.add_argument(
        "--output", type=str, default="marker_stats.csv",
        help="Output CSV path (default: marker_stats.csv)"
    )
    parser.add_argument(
        "--pattern", type=str, default="*.ome.tiff",
        help="Glob pattern for images (default: *.ome.tiff). "
             "Also try '*.ome.tif' depending on your naming."
    )
    parser.add_argument(
        "--dtype_max", type=float, default=None,
        help="Override dtype max for normalization. Auto-detected from image "
             "dtype if not set (uint16=65535, uint8=255). This value must match "
             "'marker_max_values' in your KRONOS inference config."
    )
    parser.add_argument(
        "--existing_metadata", type=str, default=None,
        help="Optional: path to existing marker_metadata.csv to compare against. "
             "Reports which of your markers are already covered vs. new."
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.rglob(args.pattern))

    if not image_paths:
        print(f"No files found matching '{args.pattern}' under {image_dir}")
        return

    print(f"Found {len(image_paths)} OME-TIFF file(s):")
    for p in image_paths:
        print(f"  {p}")

    # Accumulate moments and compute stats
    moment_store, dtype_max, processed_count, skipped_count = accumulate_moments(
        image_paths,
        args.dtype_max,
    )

    if not moment_store:
        raise RuntimeError(
            "No readable OME-TIFF files were processed. "
            "Check file integrity and try a narrower --pattern."
        )

    stats_df = compute_stats(moment_store)

    print(f"\nProcessed files: {processed_count}")
    if skipped_count > 0:
        print(f"Skipped files:   {skipped_count} (see [error] lines above)")

    print(f"\nNormalization: raw pixel / {dtype_max}")
    print(f"=> Set 'marker_max_values': {dtype_max} in your KRONOS inference config.\n")
    print("Computed stats:")
    print(stats_df[["marker_name", "marker_mean", "marker_std",
                     "n_pixels"]].to_string(index=False))

    # Compare against existing KRONOS metadata if provided
    if args.existing_metadata:
        existing = pd.read_csv(args.existing_metadata)
        existing_names = set(existing["marker_name"].str.upper())
        your_names     = set(stats_df["marker_name"])

        matched  = your_names & existing_names
        new_only = your_names - existing_names

        print(f"\n--- Comparison with {args.existing_metadata} ---")
        print(f"  Matched to KRONOS pretraining set : {len(matched)}")
        print(f"  New markers (not in pretraining)  : {len(new_only)}")

        if new_only:
            print("\n  New markers — use computed stats and assign an unused marker ID:")
            new_df = stats_df[stats_df["marker_name"].isin(new_only)][
                ["marker_name", "marker_mean", "marker_std"]
            ]
            print(new_df.to_string(index=False))

    # Save (drop diagnostic n_pixels column for clean KRONOS-ready output)
    out_path = Path(args.output)
    stats_df[["marker_name", "marker_mean", "marker_std"]].to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()