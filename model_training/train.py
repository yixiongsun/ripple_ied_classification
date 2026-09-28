"""Shared loss, training, evaluation, and checkpoint helpers for the CNN."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SharedModalityModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subject_kfold(samples: Sequence[dict], k: int = 5, seed: int = 0):
    """Split samples into deterministic folds without dividing subjects."""
    by_subject = defaultdict(list)
    for sample in samples:
        by_subject[sample["animal_id"]].append(sample)
    if not 2 <= k <= len(by_subject):
        raise ValueError("k must be between 2 and the number of subjects")

    subjects = np.asarray(sorted(by_subject), dtype=object)
    np.random.default_rng(seed).shuffle(subjects)
    folds = []
    for validation_subjects in np.array_split(subjects, k):
        validation_subjects = set(validation_subjects.tolist())
        train_samples = []
        validation_samples = []
        for subject, subject_samples in by_subject.items():
            target = validation_samples if subject in validation_subjects else train_samples
            target.extend(subject_samples)
        folds.append((train_samples, validation_samples))
    return folds


def replace_batch_norm_with_group_norm(module: nn.Module) -> None:
    """Replace BatchNorm layers in-place while preserving channel counts."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            groups = min(8, child.num_features)
            while child.num_features % groups:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            replace_batch_norm_with_group_norm(child)


# Historical public name.
replace_bn_with_gn = replace_batch_norm_with_group_norm


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor | None = None,
    *,
    temperature: float = 0.1,
    prefer_cross_subject: bool = False,
) -> torch.Tensor:
    """Compute supervised contrastive loss, excluding each anchor itself."""
    if len(embeddings) < 2:
        return embeddings.new_tensor(0.0)
    embeddings = F.normalize(embeddings, dim=1)
    similarity = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(len(embeddings), device=embeddings.device, dtype=torch.bool)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask

    if prefer_cross_subject and subjects is not None:
        preferred = positives & (subjects[:, None] != subjects[None, :])
        has_preferred = preferred.any(dim=1)
        positives = torch.where(has_preferred[:, None], preferred, positives)

    similarity -= similarity.max(dim=1, keepdim=True).values.detach()
    denominator = (torch.exp(similarity) * (~self_mask)).sum(dim=1)
    log_probability = similarity - torch.log(denominator[:, None] + 1e-8)
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)
    losses = -(positives * log_probability).sum(dim=1) / positive_count.clamp(min=1)
    return losses[valid].mean()


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight):
        ctx.weight = weight
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.weight * gradient, None


class SubjectAdversary(nn.Module):
    """Optional subject classifier used through gradient reversal in CV tests."""

    def __init__(self, embedding_dim: int, subject_count: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, subject_count),
        )

    def forward(self, embeddings: torch.Tensor, reversal_weight: float = 1.0):
        return self.network(_GradientReverse.apply(embeddings, reversal_weight))


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    contrast_weight: float = 1.0,
    noise_weight: float = 0.5,
    repulsion_weight: float = 0.5,
    noise_margin: float = 5.0,
    prefer_cross_subject: bool = False,
    subject_adversary: SubjectAdversary | None = None,
    subject_weight: float = 0.0,
    reversal_weight: float = 1.0,
) -> dict[str, float]:
    """Train one epoch with the final binary-plus-noise objective."""
    model.train()
    if subject_adversary is not None:
        subject_adversary.train()
    names = ("supervised", "contrast", "noise", "repulsion", "subject", "total")
    totals = {name: 0.0 for name in names}

    for wave, spec, global_features, labels, subject_ids in loader:
        wave = wave.to(device)
        spec = spec.to(device)
        global_features = global_features.to(device)
        labels = labels.to(device)
        subject_ids = subject_ids.to(device)
        noise_mask = labels == 2
        known_mask = ~noise_mask

        logits, embeddings, _ = model(wave, spec, global_features)
        embeddings = F.normalize(embeddings, dim=1)
        if known_mask.any():
            supervised = F.binary_cross_entropy_with_logits(
                logits[known_mask], labels[known_mask].float()
            )
            contrast = supervised_contrastive_loss(
                embeddings[known_mask],
                labels[known_mask],
                subject_ids[known_mask],
                prefer_cross_subject=prefer_cross_subject,
            )
        else:
            supervised = logits.new_tensor(0.0)
            contrast = logits.new_tensor(0.0)

        if noise_mask.any():
            noise_probability = torch.sigmoid(logits[noise_mask])
            noise = F.mse_loss(
                noise_probability, torch.full_like(noise_probability, 0.5)
            )
        else:
            noise = logits.new_tensor(0.0)

        if noise_mask.any() and known_mask.any():
            distances = torch.cdist(embeddings[noise_mask], embeddings[known_mask])
            repulsion = torch.clamp(
                noise_margin - distances.min(dim=1).values, min=0
            ).mean()
        else:
            repulsion = logits.new_tensor(0.0)

        if subject_adversary is not None and subject_weight > 0:
            subject_logits = subject_adversary(embeddings, reversal_weight)
            subject_loss = F.cross_entropy(subject_logits, subject_ids)
        else:
            subject_loss = logits.new_tensor(0.0)

        loss = (
            supervised
            + contrast_weight * contrast
            + noise_weight * noise
            + repulsion_weight * repulsion
            + subject_weight * subject_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        values = {
            "supervised": supervised,
            "contrast": contrast,
            "noise": noise,
            "repulsion": repulsion,
            "subject": subject_loss,
            "total": loss,
        }
        for name, value in values.items():
            totals[name] += float(value.detach())

    if not len(loader):
        raise ValueError("Training loader produced no batches")
    return {name: value / len(loader) for name, value in totals.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, mode: str = "both"):
    """Return probabilities, labels, subject indices, embeddings, and energies."""
    model.eval()
    probabilities = []
    labels = []
    subjects = []
    embeddings = []
    energies = []
    for wave, spec, global_features, batch_labels, subject_ids in loader:
        logits, batch_embeddings, _ = model(
            wave.to(device), spec.to(device), global_features.to(device), mode=mode
        )
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_labels.numpy())
        subjects.append(subject_ids.numpy())
        embeddings.append(batch_embeddings.cpu().numpy())
        energies.append(torch.abs(logits).cpu().numpy())
    return tuple(
        np.concatenate(parts)
        for parts in (probabilities, labels, subjects, embeddings, energies)
    )


def predict_classes(
    probabilities: np.ndarray,
    energies: np.ndarray,
    energy_threshold: float,
) -> np.ndarray:
    """Map binary probabilities and low-energy rejection to three classes."""
    predictions = (np.asarray(probabilities) > 0.5).astype(np.int64)
    predictions[np.asarray(energies) < energy_threshold] = 2
    return predictions


def load_checkpoint(path: str | Path, device: torch.device):
    """Load an existing CNN checkpoint and feature-normalization values."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SharedModalityModel(
        feat_dim=checkpoint.get("feat_dim", 64),
        n_classes=1,
        use_global_features=checkpoint.get("use_global_features", True),
    )
    replace_batch_norm_with_group_norm(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def print_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str]) -> None:
    print("\nConfusion Matrix:")
    print(" " * 15 + " ".join(f"{name:>12}" for name in class_names))
    for name, row in zip(class_names, matrix):
        print(f"{name:>15} " + " ".join(f"{value:12d}" for value in row))
