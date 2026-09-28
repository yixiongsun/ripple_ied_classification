"""Train the final CNN on the complete labeled dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .loaders import load_samples, make_balanced_loader
from .model import SharedModalityModel
from .train import replace_batch_norm_with_group_norm, set_seed, train_epoch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def train_full_dataset(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    energy_threshold: float = 3.13,
    num_workers: int = 4,
    seed: int = 0,
    device_name: str = "auto",
    overwrite: bool = False,
) -> None:
    """Run the final full-dataset configuration from the notebook."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}. "
            "Pass --overwrite or choose another --output path."
        )
    set_seed(seed)
    device = resolve_device(device_name)
    samples = load_samples(dataset_path)
    use_global_features = True

    model = SharedModalityModel(
        feat_dim=64,
        n_classes=1,
        use_global_features=use_global_features,
    )
    replace_batch_norm_with_group_norm(model)
    model.to(device)

    loader, feature_mean, feature_std = make_balanced_loader(
        samples,
        batch_size,
        use_global_features=use_global_features,
        num_workers=num_workers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        stats = train_epoch(model, loader, optimizer, device)
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"loss={stats['total']:.4f} | "
            f"supervised={stats['supervised']:.4f} | "
            f"contrast={stats['contrast']:.4f} | "
            f"noise={stats['noise']:.4f} | "
            f"repulsion={stats['repulsion']:.4f}"
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feat_mean": feature_mean,
        "feat_std": feature_std,
        "energy_threshold": energy_threshold,
        "best_epoch": epochs,
        "use_global_features": use_global_features,
        "feat_dim": 64,
    }
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset_arcsinh.pkl")
    parser.add_argument("--output", default="final_ripple_model.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--energy-threshold", type=float, default=3.13)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train_full_dataset(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        energy_threshold=args.energy_threshold,
        num_workers=args.workers,
        seed=args.seed,
        device_name=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
