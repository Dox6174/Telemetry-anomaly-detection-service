import torch
import torch.nn as nn

N_CHANNELS = 5
WINDOW = 30
INPUT_DIM = N_CHANNELS * WINDOW  # 150

class Autoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))