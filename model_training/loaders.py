"""Dataset loading and feature preparation for CNN training and inference."""

from __future__ import annotations

import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import orjson
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler


def load_samples(path: str | Path) -> list[dict]:
    """Load a serialized training dataset."""
    with Path(path).open("rb") as source:
        samples = pickle.load(source)
    if not isinstance(samples, list):
        raise TypeError(f"Expected a list of samples in {path}")
    return samples


def load_event_samples(
    event_directory: str | Path,
    animal_id: str,
    session: str,
    candidate_label: int,
    workers: int = 4,
) -> list[dict]:
    """Load normalized event JSON files in deterministic filename order."""
    paths = sorted(Path(event_directory).glob("*.json"))

    def read_event(path: Path):
        event = orjson.loads(path.read_bytes())
        match = re.search(r"(\d+)$", path.stem)
        event_num = int(match.group(1)) if match else None
        return path, event_num, event

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            loaded_events = executor.map(read_event, paths)
            loaded_events = list(loaded_events)
    else:
        loaded_events = [read_event(path) for path in paths]

    samples = []
    for fallback_num, (path, event_num, event) in enumerate(loaded_events):
        samples.append(
            {
                "animal_id": animal_id,
                "session": session,
                "window": np.asarray(event["window"], dtype=np.int64),
                "event_num": fallback_num if event_num is None else event_num,
                "file": str(path),
                "label": candidate_label,
                "wave": np.asarray(event["wave"], dtype=np.float32),
                "spec": np.asarray(event["spec"], dtype=np.float32),
                "ratio": np.asarray(event["ratio"], dtype=np.float32),
                "entropy": np.asarray(event["entropy"], dtype=np.float32),
            }
        )
    return samples


def batched_mean_coherence(
    waves: np.ndarray,
    fs: float = 1_280,
    band: tuple[float, float] = (100, 250),
    nperseg: int = 64,
    noverlap: int = 32,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return mean pairwise ripple-band coherence for each sample."""
    waves = np.asarray(waves, dtype=np.float32)
    sample_count, channel_count, time_count = waves.shape
    if time_count < nperseg:
        raise ValueError(f"T={time_count} is smaller than nperseg={nperseg}")

    step = nperseg - noverlap
    starts = np.arange(0, time_count - nperseg + 1, step)
    segments = np.stack(
        [waves[:, :, start : start + nperseg] for start in starts], axis=2
    )
    segments *= np.hanning(nperseg).astype(np.float32)[None, None, None, :]
    spectra = np.fft.rfft(segments, axis=-1)
    frequencies = np.fft.rfftfreq(nperseg, d=1 / fs)
    frequency_mask = (frequencies >= band[0]) & (frequencies <= band[1])

    pair_scores = []
    for first in range(channel_count):
        for second in range(first + 1, channel_count):
            x_first = spectra[:, first]
            x_second = spectra[:, second]
            p_first = np.mean(np.abs(x_first) ** 2, axis=1)
            p_second = np.mean(np.abs(x_second) ** 2, axis=1)
            cross_power = np.mean(x_first * np.conj(x_second), axis=1)
            coherence = np.abs(cross_power) ** 2 / (p_first * p_second + eps)
            pair_scores.append(coherence[:, frequency_mask].mean(axis=1))

    if not pair_scores:
        return np.zeros(sample_count, dtype=np.float32)
    return np.stack(pair_scores, axis=1).mean(axis=1).astype(np.float32)


def precompute_features(
    samples: Sequence[dict],
    use_global_features: bool = True,
    chunk_size: int = 4_096,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Compute coherence and entropy/ratio summary features."""
    if not use_global_features:
        return None, None, None
    if not samples:
        raise ValueError("Cannot compute features for an empty dataset")

    sample_count = len(samples)
    coherence_parts = []
    entropy_mean = np.empty(sample_count, dtype=np.float32)
    ratio_mean = np.empty(sample_count, dtype=np.float32)
    entropy_std = np.empty(sample_count, dtype=np.float32)
    ratio_std = np.empty(sample_count, dtype=np.float32)
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        waves = np.stack([samples[index]["wave"] for index in range(start, end)])
        coherence_parts.append(batched_mean_coherence(waves))
        for index in range(start, end):
            entropy = np.asarray(samples[index]["entropy"])
            ratio = np.asarray(samples[index]["ratio"])
            entropy_mean[index] = entropy.mean()
            entropy_std[index] = entropy.std()
            ratio_mean[index] = ratio.mean()
            ratio_std[index] = ratio.std()

    features = np.stack(
        [
            np.concatenate(coherence_parts),
            entropy_mean,
            ratio_mean,
            entropy_std,
            ratio_std,
        ],
        axis=1,
    ).astype(np.float32)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0) + 1e-8
    return features, feature_mean, feature_std


class LFPDataset(torch.utils.data.Dataset):
    """Torch dataset for waveform, spectrogram, and global CNN inputs."""

    def __init__(
        self,
        samples: Sequence[dict],
        features: np.ndarray | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        *,
        augment: bool = False,
        use_global_features: bool = True,
        subject_to_index: dict[str, int] | None = None,
    ):
        self.samples = samples
        self.features = features
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.augment = augment
        self.use_global_features = use_global_features
        if subject_to_index is None:
            subject_ids = sorted({sample["animal_id"] for sample in samples})
            subject_to_index = {
                subject_id: index for index, subject_id in enumerate(subject_ids)
            }
        self.subject_to_index = subject_to_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        wave = np.asarray(sample["wave"], dtype=np.float32)
        spec = np.asarray(sample["spec"], dtype=np.float32)
        if self.augment:
            wave, spec = self._augment(wave, spec)
        if self.use_global_features:
            feature = (self.features[index] - self.feature_mean) / self.feature_std
            feature = np.clip(feature, -5, 5).astype(np.float32)
        else:
            feature = np.empty(0, dtype=np.float32)
        return (
            torch.from_numpy(wave),
            torch.from_numpy(spec),
            torch.from_numpy(feature),
            torch.tensor(sample["label"], dtype=torch.long),
            torch.tensor(self.subject_to_index[sample["animal_id"]], dtype=torch.long),
        )

    @staticmethod
    def _augment(wave: np.ndarray, spec: np.ndarray):
        wave = wave.copy()
        spec = spec.copy()
        channel_count, time_count = wave.shape
        if np.random.random() < 0.3:
            wave += np.random.normal(
                0, 0.05, size=(channel_count, time_count)
            ).astype(np.float32)
            spec += np.random.normal(0, 0.02, size=spec.shape).astype(np.float32)
        if np.random.random() < 0.5:
            permutation = np.random.permutation(channel_count)
            wave = wave[permutation]
            spec = spec[permutation]
        if np.random.random() < 0.3:
            shift = np.random.randint(-5, 6)
            wave = np.roll(wave, shift, axis=1)
            spec = np.roll(spec, shift, axis=2)
        return wave.astype(np.float32), spec.astype(np.float32)


class SubjectClassBalancedBatchSampler(Sampler[list[int]]):
    """Create batches balanced across both subjects and event classes."""

    def __init__(
        self,
        samples: Sequence[dict],
        batch_size: int,
        subjects_per_batch: int = 8,
        classes: tuple[int, ...] = (0, 1, 2),
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.subjects_per_batch = subjects_per_batch
        self.classes = classes
        self.rng = np.random.default_rng(seed)
        self.subject_class_indices: dict[str, dict[int, list[int]]] = {}
        for index, sample in enumerate(samples):
            by_class = self.subject_class_indices.setdefault(sample["animal_id"], {})
            by_class.setdefault(sample["label"], []).append(index)
        self.subjects = np.asarray(list(self.subject_class_indices))
        self.batch_count = len(samples) // batch_size

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self):
        class_subjects = {
            label: [
                subject
                for subject in self.subjects
                if self.subject_class_indices[subject].get(label)
            ]
            for label in self.classes
        }
        missing_classes = [label for label, subjects in class_subjects.items() if not subjects]
        if missing_classes:
            raise ValueError(f"No samples are available for classes {missing_classes}")
        for _ in range(self.batch_count):
            batch = []
            selected_subjects = self.rng.choice(
                self.subjects,
                size=min(self.subjects_per_batch, len(self.subjects)),
                replace=False,
            )
            per_subject = max(1, self.batch_size // len(selected_subjects))
            per_class = max(1, per_subject // len(self.classes))
            for subject in selected_subjects:
                for label in self.classes:
                    indices = self.subject_class_indices[subject].get(label)
                    if indices:
                        selected = self.rng.choice(
                            indices,
                            size=per_class,
                            replace=len(indices) < per_class,
                        )
                        batch.extend(selected.tolist())
            while len(batch) < self.batch_size:
                label = int(self.rng.choice(self.classes))
                candidates = class_subjects[label]
                if not candidates:
                    continue
                subject = self.rng.choice(candidates)
                batch.append(self.rng.choice(self.subject_class_indices[subject][label]))
            self.rng.shuffle(batch)
            yield batch[: self.batch_size]


def _loader_options(num_workers: int) -> dict:
    return {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }


def make_subject_class_balanced_loader(
    samples: Sequence[dict],
    batch_size: int,
    *,
    use_global_features: bool = True,
    num_workers: int = 4,
    subjects_per_batch: int = 8,
    classes: tuple[int, ...] = (0, 1, 2),
    seed: int = 0,
):
    features, feature_mean, feature_std = precompute_features(samples, use_global_features)
    subject_ids = sorted({sample["animal_id"] for sample in samples})
    subject_to_index = {subject_id: index for index, subject_id in enumerate(subject_ids)}
    dataset = LFPDataset(
        samples,
        features,
        feature_mean,
        feature_std,
        use_global_features=use_global_features,
        subject_to_index=subject_to_index,
    )
    sampler = SubjectClassBalancedBatchSampler(
        samples, batch_size, subjects_per_batch, classes, seed
    )
    loader = DataLoader(dataset, batch_sampler=sampler, **_loader_options(num_workers))
    return loader, feature_mean, feature_std, subject_to_index


def make_balanced_loader(
    samples: Sequence[dict],
    batch_size: int,
    *,
    use_global_features: bool = True,
    num_workers: int = 4,
):
    features, feature_mean, feature_std = precompute_features(samples, use_global_features)
    dataset = LFPDataset(
        samples,
        features,
        feature_mean,
        feature_std,
        augment=True,
        use_global_features=use_global_features,
    )
    labels = np.asarray([sample["label"] for sample in samples])
    counts = np.bincount(labels, minlength=3)
    class_weights = 1.0 / (counts + 1e-6)
    sampler = WeightedRandomSampler(
        torch.as_tensor(class_weights[labels], dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        **_loader_options(num_workers),
    )
    return loader, feature_mean, feature_std


def make_evaluation_loader(
    samples: Sequence[dict],
    batch_size: int,
    *,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
    use_global_features: bool = True,
    num_workers: int = 4,
):
    features, _, _ = precompute_features(samples, use_global_features)
    dataset = LFPDataset(
        samples,
        features,
        feature_mean,
        feature_std,
        use_global_features=use_global_features,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **_loader_options(num_workers),
    )


# Compatibility name used by the historical notebooks.
make_val_loader = make_evaluation_loader
