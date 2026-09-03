#!/usr/bin/env python3
"""Safely extract the six Baidu archives into the portable runtime layout."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import tarfile


ARCHIVES = (
    ("data/ADFTD.tar.gz", "data/eeg"),
    ("data/TDBRAIN.tar.gz", "data/eeg"),
    ("data/APAVA.tar.gz", "data/eeg"),
    ("data/Shimmer10.tar.gz", "data/wearable"),
    ("data/PADS11.tar.gz", "data/wearable"),
    (
        "checkpoints/TimeMosaic_selector_only_seed42_43_44_checkpoints.tar.gz",
        "reference_results",
    ),
)


def validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {member.name}")
    if member.issym() or member.islnk() or member.isdev():
        raise ValueError(f"links/devices are not allowed: {member.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baidu-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    arguments = parser.parse_args()
    baidu_root = arguments.baidu_root.expanduser().resolve()
    runtime_root = arguments.runtime_root.expanduser().resolve()
    if not baidu_root.is_dir():
        parser.error(f"Baidu root does not exist: {baidu_root}")
    runtime_root.mkdir(parents=True, exist_ok=True)

    for relative_archive, relative_destination in ARCHIVES:
        archive = baidu_root / relative_archive
        destination = runtime_root / relative_destination
        if not archive.is_file():
            raise SystemExit(f"missing archive: {archive}")
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive.name} -> {destination}", flush=True)
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            for member in members:
                validate_member(member)
            # Every supplied archive was audited to contain regular files and
            # directories only. Numeric ownership is deliberately not restored.
            handle.extractall(destination, members=members, filter="data")

    expected = (
        runtime_root / "data/eeg/ADFTD/Label/label.npy",
        runtime_root / "data/eeg/TDBRAIN/Label/label.npy",
        runtime_root / "data/eeg/APAVA/Label/label.npy",
        runtime_root
        / "data/wearable/Shimmer_10_session10_AFC/Meta/subject_map.csv",
        runtime_root
        / "data/wearable/PADS_11_task08_TouchIndex/Meta/subject_map.csv",
        runtime_root
        / "reference_results/adftd/timemosaic/seed_42/metrics.json",
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise SystemExit("extraction incomplete:\n" + "\n".join(missing))
    print(f"Prepared runtime assets at {runtime_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
