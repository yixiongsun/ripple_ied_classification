"""Choose representative ripple and IED candidates for manual labeling.

This is the Python replacement for ``kmeans_selection.ipynb``.  Candidate
JSON files are expected to use the final event-classification schema:
``wave``, ``spec``, ``ratio``, and ``entropy``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


CSV_COLUMNS = ("subject_id", "session_id", "image_path", "label")


def prepare_features(
    waveforms: np.ndarray,
    spectrograms: np.ndarray,
    additional_features: np.ndarray,
    *,
    use_pca: bool = True,
    pca_dimensions: int = 100,
) -> np.ndarray:
    """Standardize and combine the notebook's three feature groups."""
    sample_count = len(waveforms)
    if sample_count == 0:
        return np.empty((0, 0), dtype=np.float32)

    waveform_features = np.asarray(waveforms, dtype=np.float32).reshape(sample_count, -1)
    spectrogram_features = np.asarray(spectrograms, dtype=np.float32).reshape(sample_count, -1)
    additional_features = np.asarray(additional_features, dtype=np.float32).reshape(
        sample_count, -1
    )

    waveform_features = StandardScaler(copy=False).fit_transform(waveform_features)
    spectrogram_features = StandardScaler(copy=False).fit_transform(spectrogram_features)
    additional_features = StandardScaler(copy=False).fit_transform(additional_features)

    # Preserve the relative weights used by the notebook.
    waveform_features *= 0.5
    features = np.concatenate(
        (waveform_features, spectrogram_features, additional_features), axis=1
    )

    if use_pca:
        dimensions = min(pca_dimensions, features.shape[0], features.shape[1])
        features = PCA(n_components=dimensions, random_state=0).fit_transform(features)
    return features


def select_kmeans_representations(
    features: np.ndarray,
    count: int = 100,
    *,
    random_state: int = 0,
) -> np.ndarray:
    """Return the sample nearest each K-means cluster center."""
    features = np.asarray(features)
    if count < 1:
        raise ValueError("count must be at least 1")
    if len(features) <= count:
        return np.arange(len(features), dtype=np.int64)

    kmeans = KMeans(n_clusters=count, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    selected = []
    for cluster_index, center in enumerate(kmeans.cluster_centers_):
        members = np.flatnonzero(cluster_labels == cluster_index)
        distances = np.linalg.norm(features[members] - center, axis=1)
        selected.append(int(members[np.argmin(distances)]))
    return np.asarray(selected, dtype=np.int64)


def _event_arrays(record: dict, source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        wave = np.asarray(record["wave"], dtype=np.float32)
        spec = np.asarray(record["spec"], dtype=np.float32)
        ratio = np.asarray(record["ratio"], dtype=np.float32)
        entropy = np.asarray(record["entropy"], dtype=np.float32)
    except KeyError as error:
        raise ValueError(f"{source} does not use the final event JSON schema") from error

    if ratio.shape != entropy.shape:
        raise ValueError(f"ratio and entropy shapes differ in {source}")
    # This matches np.array([ratio, entropy]).T from the notebook.
    additional = np.column_stack((ratio, entropy))
    return wave, spec, additional


def load_candidate_features(
    json_files: Iterable[str | Path],
    *,
    show_progress: bool = True,
) -> tuple[list[Path], np.ndarray, np.ndarray, np.ndarray]:
    """Load the features used for selection from candidate JSON files."""
    paths = sorted((Path(path) for path in json_files), key=lambda path: path.name)
    if not paths:
        return (
            paths,
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    missing_images = [
        path.with_suffix(".jpg")
        for path in paths
        if not path.with_suffix(".jpg").is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"Candidate image is missing: {missing_images[0]}")

    with paths[0].open("r", encoding="utf-8") as input_file:
        first_wave, first_spec, first_additional = _event_arrays(
            json.load(input_file), paths[0]
        )
    waves = np.empty((len(paths), *first_wave.shape), dtype=np.float32)
    spectra = np.empty((len(paths), *first_spec.shape), dtype=np.float32)
    additional = np.empty((len(paths), *first_additional.shape), dtype=np.float32)
    waves[0], spectra[0], additional[0] = first_wave, first_spec, first_additional

    iterator = tqdm(paths[1:], desc="load candidates", disable=not show_progress)
    for index, path in enumerate(iterator, start=1):
        with path.open("r", encoding="utf-8") as input_file:
            wave, spec, extra = _event_arrays(json.load(input_file), path)
        try:
            waves[index], spectra[index], additional[index] = wave, spec, extra
        except ValueError as error:
            raise ValueError(f"Feature shapes in {path} differ from the first candidate") from error
    return paths, waves, spectra, additional


def select_event_candidates(
    event_directory: str | Path,
    subject_id: str,
    session_id: str,
    *,
    count: int = 100,
    random_state: int = 0,
    show_progress: bool = True,
) -> list[dict[str, str]]:
    """Select candidates of one event type and return labeling CSV rows."""
    event_directory = Path(event_directory)
    paths, waves, spectra, additional = load_candidate_features(
        event_directory.glob("*.json"), show_progress=show_progress
    )
    if not paths:
        return []

    if len(paths) <= count:
        selected_indices = np.arange(len(paths), dtype=np.int64)
    else:
        features = prepare_features(waves, spectra, additional)
        del waves, spectra, additional
        selected_indices = select_kmeans_representations(
            features, count=count, random_state=random_state
        )

    rows = []
    for index in selected_indices:
        # Use the actual selected filename. The notebook reconstructed a name
        # from the list position, which was wrong when event numbers had gaps.
        image_path = paths[int(index)].with_suffix(".jpg")
        rows.append(
            {
                "subject_id": str(subject_id),
                "session_id": str(session_id),
                "image_path": str(image_path),
                "label": "",
            }
        )
    return rows


def select_session_candidates(
    session_directory: str | Path,
    subject_id: str,
    session_id: str,
    *,
    count_per_event_type: int = 100,
    random_state: int = 0,
    show_progress: bool = True,
) -> list[dict[str, str]]:
    """Select both ripple and IED candidates from one recording session."""
    session_directory = Path(session_directory)
    rows = []
    for event_directory in ("ripples", "ieds"):
        rows.extend(
            select_event_candidates(
                session_directory / event_directory,
                subject_id,
                session_id,
                count=count_per_event_type,
                random_state=random_state,
                show_progress=show_progress,
            )
        )
    return rows


def write_label_csv(rows: Iterable[dict[str, str]], output_path: str | Path) -> None:
    """Write rows using the exact column order expected by ``label_app.py``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
