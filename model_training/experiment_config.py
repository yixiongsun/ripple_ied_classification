"""Frozen settings shared by model-selection experiments."""

from __future__ import annotations

from .compare_losses import LossConfig


FROZEN_ARCHITECTURE = "compact_wave_no_temporal"

FROZEN_LOSS = LossConfig(
    name="frozen_binary_rejection",
    contrast_weight=0.1,
    contrast_temperature=0.1,
    noise_weight=0.5,
    repulsion_weight=0.1,
    noise_margin=0.5,
)

FROZEN_BATCH_SIZE = 32
FROZEN_SUBJECTS_PER_BATCH = 8
FROZEN_EPOCHS = 40
FROZEN_CALIBRATION_FRACTION = 0.2
FROZEN_LEARNING_RATE = 1e-3
FROZEN_WEIGHT_DECAY = 0.0
FROZEN_SCHEDULE = "constant"
FROZEN_DROPOUT = 0.0
FROZEN_MAX_GRAD_NORM = None
FROZEN_FEATURE_DIM = 64
FROZEN_THRESHOLD_SELECTION = "maximize calibration macro F1"
