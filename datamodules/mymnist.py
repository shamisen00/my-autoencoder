import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

class MyMNIST(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, data_dir: str = "./"):
        super().__init__()
        dataset = MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)