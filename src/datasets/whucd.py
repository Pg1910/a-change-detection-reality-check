import glob
import os
from typing import Optional, Callable

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, random_split
from lightning import LightningDataModule
from kornia.augmentation.container import AugmentationSequential


class WHUCD(Dataset):
    splits = ["train", "test"]

    def __init__(self, root: str = "data", split: str = "train", transforms: Optional[Callable] = None) -> None:
        assert split in self.splits
        self.root = root
        self.split = split
        self.transforms = transforms
        self.files = self._load_files()

    def _load_files(self) -> list[dict[str, str]]:
        image1_paths = sorted(glob.glob(os.path.join(self.root, "2012", self.split, "*.tif")))
        image2_paths = sorted(glob.glob(os.path.join(self.root, "2016", self.split, "*.tif")))
        mask_paths = sorted(glob.glob(os.path.join(self.root, "change_label", self.split, "*.tif")))

        return [{"image1": a, "image2": b, "mask": m} for a, b, m in zip(image1_paths, image2_paths, mask_paths)]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        paths = self.files[idx]
        image1 = self._load_image(paths["image1"])
        image2 = self._load_image(paths["image2"])
        mask = self._load_mask(paths["mask"])
        sample = {"image1": image1, "image2": image2, "mask": mask}
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def _load_image(self, path: str) -> Tensor:
        img = Image.open(path).convert("RGB")
        array = np.array(img).astype(np.float32) / 255.0
        return torch.tensor(array).permute(2, 0, 1)

    def _load_mask(self, path: str) -> Tensor:
        img = Image.open(path).convert("L")
        array = np.array(img)
        binary_mask = (array > 0).astype(np.int64)
        return torch.tensor(binary_mask)

    def plot(self, sample: dict[str, Tensor], show_titles=True, suptitle=None) -> Figure:
        ncols = 3 + int("prediction" in sample)
        fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

        axs[0].imshow(sample["image1"].permute(1, 2, 0).numpy())
        axs[1].imshow(sample["image2"].permute(1, 2, 0).numpy())
        axs[2].imshow(sample["mask"].numpy(), cmap="gray")

        if "prediction" in sample:
            axs[3].imshow(sample["prediction"].numpy(), cmap="gray")
            if show_titles:
                axs[3].set_title("Prediction")

        for i, key in enumerate(["Image 1", "Image 2", "Mask"] + (["Prediction"] if "prediction" in sample else [])):
            axs[i].axis("off")
            if show_titles:
                axs[i].set_title(key)

        if suptitle:
            fig.suptitle(suptitle)

        return fig


class WHUCDDataModule(LightningDataModule):
    def __init__(self, root: str = "data", batch_size: int = 8, patch_size: int = 256, num_workers: int = 2):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        full_dataset = WHUCD(root=self.root, split="train", transforms=self.train_transform())
        val_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.test_dataset = WHUCD(root=self.root, split="test", transforms=self.test_transform())

    def train_transform(self):
        return AugmentationSequential(
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomResizedCrop(self.patch_size, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            data_keys=["image1", "image2", "mask"]
        )

    def test_transform(self):
        return AugmentationSequential(
            data_keys=["image1", "image2", "mask"]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
