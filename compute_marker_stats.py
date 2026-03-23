"""
compute_marker_stats.py
-----------------------
Compute per-marker mean and std (in 0-1 range) from COMET OME-TIFF images
for use in KRONOS marker_metadata.csv.

Normalization strategy:
  1. Per-image, per-channel: clip at the 99th percentile, scale to [0, 1]
  2. Pool all normalized pixels across all images and samples
  3. Compute global mean and std on the pooled normalized pixels

Usage:
    python compute_marker_stats.py \
        --image_dir /path/to/ome_tiffs \
        --output marker_metadata_new.csv \
        --pattern "*.ome.tiff"
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


# ---------------------------------------------------------------------------
# OME-TIFF helpers
# ---------------------------------------------------------------------------

def parse_marker_names_from_ome(tif: tifffile.TiffFile) -> list[str]:
    """
    Extract ordered channel names from the OME-XML metadata embedded in a
    COMET OME-TIFF. Returns a list aligned with the channel axis of the array.
    Falls back to generic names (Ch_0, Ch_1, ...) if parsing fails.
    """
    try:
        ome_xml = tif.ome_metadata          # full OME-XML string
        root = ET.fromstring(ome_xml)

        # OME namespace varies by version — find it dynamically
        ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
        ns_prefix = f"{{{ns}}}" if ns else ""

        # Walk to Image > Pixels > Channel
        image = root.find(f".//{ns_prefix}Image")
        pixels = image.find(f"{ns_prefix}Pixels")
        channels = pixels.findall(f"{ns_prefix}Channel")

        names = []
        for ch in channels:
            name = ch.get("Name") or ch.get("Fluor") or ch.get("ID", "")
            names.append(name.upper().strip())
        return names

    except Exception as e:
        print(f"  [warn] Could not parse OME-XML channel names: {e}")
        return []


def read_ome_tiff(path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Read a COMET OME-TIFF.
    Returns:
        image  : float32 array of shape (C, H, W)
        markers: list of marker/channel names, length C
    """
    with tifffile.TiffFile(path) as tif:
        marker_names = parse_marker_names_from_ome(tif)

        # tifffile returns (C, H, W) for most OME-TIFFs;
        # squeeze out any singleton dimensions (Z, T) just in case
        image = tif.asarray()

    # Ensure float32
    image = image.astype(np.float32)

    # Handle dimension ordering — we want (C, H, W)
    if image.ndim == 2:
        image = image[np.newaxis, ...]       # single channel
    elif image.ndim == 3:
        pass                                  # already (C, H, W)
    elif image.ndim == 4:
        # Could be (Z, C, H, W) or (T, C, H, W) — take first Z/T slice
        image = image[0]
    elif image.ndim == 5:
        image = image[0, 0]                   # (T, Z, C, H, W) → (C, H, W)

    n_channels = image.shape[0]

    # Fall back to generic names if parsing failed or count mismatches
    if len(marker_names) != n_channels:
        print(f"  [warn] Channel name count ({len(marker_names)}) != "
              f"image channels ({n_channels}). Using generic names.")
        marker_names = [f"CH_{i:03d}" for i in range(n_channels)]

    return image, marker_names


# ---------------------------------------------------------------------------
# Normalization and stats
# ---------------------------------------------------------------------------

def normalize_channel(channel: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Clip at `percentile` then min-max scale to [0, 1].
    channel: 2-D float32 array (H, W)
    """
    p_high = np.percentile(channel, percentile)
    if p_high < 1e-6:
        # Blank / empty channel — return zeros
        return np.zeros_like(channel)
    clipped = np.clip(channel, 0.0, p_high)
    return clipped / p_high


def accumulate_pixels(
    image_paths: list[Path],
    percentile: float = 99.0,
) -> dict[str, np.ndarray]:
    """
    Iterate over all OME-TIFFs, normalize each channel, and accumulate
    all pixel values per marker name.

    Returns a dict: { marker_name -> 1-D float32 array of all pixel values }
    """
    pixel_store: dict[str, list[np.ndarray]] = {}

    for path in tqdm(image_paths, desc="Reading images"):
        try:
            image, marker_names = read_ome_tiff(path)
        except Exception as e:
            print(f"  [error] Skipping {path.name}: {e}")
            continue

        for ch_idx, marker in enumerate(marker_names):
            channel = image[ch_idx]                        # (H, W)
            normed  = normalize_channel(channel, percentile)
            flat    = normed.flatten()

            if marker not in pixel_store:
                pixel_store[marker] = []
            pixel_store[marker].append(flat)

        # Free memory immediately
        del image

    # Concatenate per marker
    return {
        marker: np.concatenate(arrays)
        for marker, arrays in pixel_store.items()
    }


def compute_stats(pixel_store: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compute mean and std for each marker from pooled normalized pixels.
    """
    rows = []
    for marker, pixels in sorted(pixel_store.items()):
        rows.append({
            "marker_name": marker,
            "marker_mean": round(float(np.mean(pixels)), 6),
            "marker_std":  round(float(np.std(pixels)),  6),
            "n_pixels":    len(pixels),
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
             "Also try '*.ome.tif' depending on your naming convention."
    )
    parser.add_argument(
        "--percentile", type=float, default=99.0,
        help="Percentile for clipping before normalization (default: 99)"
    )
    parser.add_argument(
        "--existing_metadata", type=str, default=None,
        help="Optional: path to existing marker_metadata.csv to merge results into"
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.rglob(args.pattern))

    if not image_paths:
        print(f"No files found matching '{args.pattern}' under {image_dir}")
        return

    print(f"Found {len(image_paths)} OME-TIFF file(s)")
    for p in image_paths:
        print(f"  {p}")

    # Accumulate pixels
    pixel_store = accumulate_pixels(image_paths, percentile=args.percentile)

    # Compute stats
    stats_df = compute_stats(pixel_store)
    print("\nComputed stats:")
    print(stats_df.to_string(index=False))

    # Optionally merge with existing marker_metadata.csv
    if args.existing_metadata:
        existing = pd.read_csv(args.existing_metadata)
        # Only keep markers NOT already in the existing metadata
        new_markers = stats_df[~stats_df["marker_name"].isin(existing["marker_name"])]
        if len(new_markers) > 0:
            print(f"\n{len(new_markers)} new marker(s) not in existing metadata:")
            print(new_markers[["marker_name", "marker_mean", "marker_std"]].to_string(index=False))
        else:
            print("\nAll markers already exist in the provided metadata file.")

    # Save — drop the diagnostic n_pixels column from the KRONOS-facing output
    out_path = Path(args.output)
    stats_df[["marker_name", "marker_mean", "marker_std"]].to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()