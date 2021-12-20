# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MNIST autoencoder example.

To run: python autoencoder.py
"""
from typing import Optional, Tuple

import torch
import pytorch_lightning as pl
import torchvision
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from models.lite_autoencoder import LitAutoEncoder
from datamodules.mymnist import MyMNIST


class ImageSampler(pl.callbacks.Callback):
    def __init__(
        self,
        num_samples: int = 16,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def _to_grid(self, images):
        return torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        images, _ = next(iter(DataLoader(trainer.datamodule.mnist_val, batch_size=self.num_samples)))
        images_flattened = images.view(images.size(0), -1)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images_generated = pl_module(images_flattened.to(pl_module.device))
            pl_module.train()

        if trainer.current_epoch == 0:
            save_image(self._to_grid(images), f"grid_ori_{trainer.current_epoch}.png")
        save_image(self._to_grid(images_generated.reshape(images.shape)), f"grid_generated_{trainer.current_epoch}.png")

@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.setting)
    print(cfg.datasets)

    early_stop_callback = instantiate(cfg.callbacks.EarlyStopping)
    trainer = Trainer(gpus=cfg.setting.params.gpus, max_epochs=cfg.setting.params.max_epochs, callbacks=[early_stop_callback,ImageSampler()])

    mnist = instantiate(cfg.datasets.MNIST)
    model = instantiate(cfg.model.lite_autoencoder)

    trainer.fit(model, mnist)
    trainer.test(model, mnist)


if __name__ == "__main__":
    main()
