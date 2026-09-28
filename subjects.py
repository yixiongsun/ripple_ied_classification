"""Resolve subject metadata for the dataset and inference pipelines."""

import json
from pathlib import Path

from settings import DISK, SUBJECT_DIRECTORY


def get_subjects():
    """Return the available subject IDs in deterministic order."""
    return sorted(path.stem for path in Path(SUBJECT_DIRECTORY).glob("*.json"))


def load(subject_id):
    """Return the recording location and LFP channels for one subject."""
    metadata_path = Path(SUBJECT_DIRECTORY) / f"{subject_id}.json"
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    return {
        "olm": {
            "directory": str(Path(DISK) / metadata["dataOlmDirectory"]),
            "lfp_channels": metadata["hpcchannels"],
        }
    }
