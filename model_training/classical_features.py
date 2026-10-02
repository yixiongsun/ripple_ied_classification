"""Feature assembly and leakage-safe preprocessing for classical baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .loaders import precompute_features


GLOBAL_FEATURE_NAMES = (
    "ripple_band_coherence",
    "entropy_mean",
    "ratio_mean",
    "entropy_std",
    "ratio_std",
)


def flatten_waveforms(samples: Sequence[dict]) -> np.ndarray:
    """Return C-order flattened waveforms after validating a common shape."""
    if not samples:
        raise ValueError("Cannot assemble features for an empty sample sequence")
    waves = [np.asarray(sample["wave"], dtype=np.float32) for sample in samples]
    expected_shape = waves[0].shape
    if len(expected_shape) != 2:
        raise ValueError(f"Expected 2-D waveforms, received shape {expected_shape}")
    for index, wave in enumerate(waves):
        if wave.shape != expected_shape:
            raise ValueError(
                f"Waveform {index} has shape {wave.shape}; expected {expected_shape}"
            )
        if not np.isfinite(wave).all():
            raise ValueError(f"Waveform {index} contains non-finite values")
    return np.stack(waves).reshape(len(waves), -1).astype(np.float32, copy=False)


def extract_global_features(samples: Sequence[dict]) -> np.ndarray:
    """Extract the same five global features used by the compact neural model."""
    features, _, _ = precompute_features(samples, use_global_features=True)
    if features is None or features.shape != (len(samples), len(GLOBAL_FEATURE_NAMES)):
        raise ValueError(
            "Global feature extraction returned an unexpected shape: "
            f"{None if features is None else features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("Global features contain non-finite values")
    return np.asarray(features, dtype=np.float32)


@dataclass(frozen=True)
class ClassicalFeatureMatrix:
    """Precomputed, stateless event features and their column boundaries."""

    values: np.ndarray
    waveform_shape: tuple[int, int]
    waveform_feature_count: int

    @property
    def global_feature_count(self) -> int:
        return self.values.shape[1] - self.waveform_feature_count


def assemble_feature_matrix(samples: Sequence[dict]) -> ClassicalFeatureMatrix:
    """Compute waveform and global inputs once, preserving sample order."""
    waveform = flatten_waveforms(samples)
    global_features = extract_global_features(samples)
    values = np.concatenate((waveform, global_features), axis=1).astype(
        np.float32, copy=False
    )
    return ClassicalFeatureMatrix(
        values=values,
        waveform_shape=tuple(np.asarray(samples[0]["wave"]).shape),
        waveform_feature_count=waveform.shape[1],
    )


class ClassicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that assembles raw event dictionaries."""

    def fit(self, X: Sequence[dict], y=None):
        matrix = assemble_feature_matrix(X)
        self.waveform_shape_ = matrix.waveform_shape
        self.waveform_feature_count_ = matrix.waveform_feature_count
        self.n_features_out_ = matrix.values.shape[1]
        return self

    def transform(self, X: Sequence[dict]) -> np.ndarray:
        matrix = assemble_feature_matrix(X)
        if hasattr(self, "waveform_shape_") and matrix.waveform_shape != self.waveform_shape_:
            raise ValueError(
                f"Waveform shape changed from {self.waveform_shape_} to "
                f"{matrix.waveform_shape}"
            )
        return matrix.values

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "waveform_feature_count_"):
            raise RuntimeError("Transformer must be fitted before requesting names")
        waveform_names = [
            f"waveform_{index}" for index in range(self.waveform_feature_count_)
        ]
        return np.asarray([*waveform_names, *GLOBAL_FEATURE_NAMES], dtype=object)


def build_preprocessor(
    waveform_feature_count: int,
    pca_components: int | None,
    *,
    random_state: int,
) -> ColumnTransformer:
    """Build training-only waveform PCA and global-feature scaling."""
    if waveform_feature_count < 1:
        raise ValueError("waveform_feature_count must be positive")
    waveform_steps: list[tuple[str, object]] = [("scale", StandardScaler())]
    if pca_components is not None:
        if pca_components < 1:
            raise ValueError("pca_components must be positive or None")
        waveform_steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    svd_solver="randomized",
                    random_state=random_state,
                ),
            )
        )
    return ColumnTransformer(
        transformers=(
            (
                "waveform",
                Pipeline(waveform_steps),
                slice(0, waveform_feature_count),
            ),
            ("global", StandardScaler(), slice(waveform_feature_count, None)),
        ),
        remainder="drop",
        sparse_threshold=0.0,
    )

