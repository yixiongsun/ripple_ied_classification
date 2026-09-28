import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class WaveAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # 128 → 64
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 → 32
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 32, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 32),
            nn.Unflatten(1, (64, 32)),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 → 64
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # 64 → 128
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 3, kernel_size=3, padding=1)
        )

    def forward(self, x, noise_std=0.1):
        if self.training:
            x_noisy = x + noise_std * torch.randn_like(x)
        else:
            x_noisy = x

        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return x_hat



class WaveDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]["wave"], dtype=torch.float32)