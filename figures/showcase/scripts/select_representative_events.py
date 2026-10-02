"""Select reproducible ripple, IED, and ambiguous/noise event examples."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "figures" / "showcase" / "data" / "representative_events.json"
LABELS = {0: "ripple", 1: "ied", 2: "noise"}
# The automatic centroid example was morphologically central but visually weak.
# This event was chosen from the central 35% shortlist after checking that the
# oscillatory burst is centered in the labelled window on multiple channels.
AUDITED_RIPPLE_INDEX = 1021


def morphology_vector(sample: dict) -> np.ndarray:
    """Create a compact waveform + spectrum vector without fitting a model."""
    wave = np.asarray(sample["wave"], dtype=np.float32)
    wave_scale = np.std(wave, axis=1, keepdims=True)
    wave = (wave - np.mean(wave, axis=1, keepdims=True)) / np.maximum(wave_scale, 1e-6)

    spectrum = np.mean(np.asarray(sample["spec"], dtype=np.float32), axis=0)
    spectrum = spectrum.reshape(10, 10, 16, 8).mean(axis=(1, 3))
    spectrum = (spectrum - spectrum.mean()) / max(float(spectrum.std()), 1e-6)
    return np.concatenate((wave.reshape(-1), spectrum.reshape(-1)))


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    dataset_path = ROOT / "dataset_arcsinh.pkl"
    with dataset_path.open("rb") as stream:
        samples = pickle.load(stream)

    features = np.stack([morphology_vector(sample) for sample in samples])
    labels = np.asarray([int(sample["label"]) for sample in samples])
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    standardized = (features - mean) / np.maximum(scale, 1e-6)
    centroids = {label: standardized[labels == label].mean(axis=0) for label in LABELS}

    selected: list[tuple[int, str, float]] = []
    used_subjects: set[str] = set()

    ripple_candidates = np.flatnonzero(labels == 0)
    ripple_distances = np.linalg.norm(standardized[ripple_candidates] - centroids[0], axis=1)
    ripple_position = int(np.flatnonzero(ripple_candidates == AUDITED_RIPPLE_INDEX)[0])
    if ripple_distances[ripple_position] > np.quantile(ripple_distances, 0.35):
        raise ValueError("Audited ripple no longer belongs to the central 35% shortlist")
    ripple_subject = str(samples[AUDITED_RIPPLE_INDEX]["animal_id"])
    selected.append(
        (
            AUDITED_RIPPLE_INDEX,
            "central-class shortlist; visually audited for a centered multichannel oscillatory burst",
            float(ripple_distances[ripple_position]),
        )
    )
    used_subjects.add(ripple_subject)

    for label in (1,):
        candidates = np.flatnonzero(labels == label)
        distances = np.linalg.norm(standardized[candidates] - centroids[label], axis=1)
        for position in np.argsort(distances):
            index = int(candidates[position])
            subject = str(samples[index]["animal_id"])
            if subject not in used_subjects:
                selected.append((index, "nearest class centroid", float(distances[position])))
                used_subjects.add(subject)
                break

    midpoint = (centroids[0] + centroids[1]) / 2
    noise_candidates = np.flatnonzero(labels == 2)
    noise_distances = np.linalg.norm(standardized[noise_candidates] - midpoint, axis=1)
    for position in np.argsort(noise_distances):
        index = int(noise_candidates[position])
        subject = str(samples[index]["animal_id"])
        if subject not in used_subjects:
            selected.append((index, "noise nearest ripple–IED midpoint", float(noise_distances[position])))
            used_subjects.add(subject)
            break

    subject_aliases = {
        subject: f"Subject {position + 1:02d}"
        for position, subject in enumerate(sorted({str(sample["animal_id"]) for sample in samples}))
    }
    records = []
    for index, rule, distance in selected:
        sample = samples[index]
        records.append(
            {
                "dataset_index": index,
                "class": LABELS[int(sample["label"])],
                "display_subject": subject_aliases[str(sample["animal_id"])],
                "source_subject": str(sample["animal_id"]),
                "session": str(sample["session"]),
                "source_file_name": Path(str(sample["file"])).name,
                "selection_rule": rule,
                "feature_distance": distance,
            }
        )

    payload = {
        "schema_version": 1,
        "source": "dataset_arcsinh.pkl",
        "source_sha256": source_hash(dataset_path),
        "analysis_status": "New representative-selection analysis; not part of Phase 9 cross-validation.",
        "feature_space": "Per-channel normalized waveform plus channel-mean 10×16 downsampled spectrogram; features standardized across all events.",
        "events": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    for record in records:
        print(record["class"], record["display_subject"], record["dataset_index"], f"distance={record['feature_distance']:.3f}")


if __name__ == "__main__":
    main()
