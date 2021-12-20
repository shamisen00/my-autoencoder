import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 256))
        self.decoder = nn.Sequential(nn.Linear(256, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 28 * 28))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _prepare_batch(self, batch):
        x, _ = batch
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(x, self(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss