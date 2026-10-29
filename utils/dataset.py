import numpy as np
import torch

from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torch.utils.data import random_split

from .config import load_config


class PetTransform:
    def __call__(self, image, mask, image_size):

        image = F.resize(image, image_size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, image_size, interpolation=InterpolationMode.NEAREST)
        image = F.to_tensor(image)

        mask = np.array(mask)
        mask = (mask != 2).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def load_dataset():
    config = load_config()
    image_size = config["dataset"]["image_size"]
    image_root = config["dataset"]["root"]
    val_ratio = config["dataset"]["val_ratio"]
    seed = config["training"]["seed"]

    dataset = OxfordIIITPet(
        root=image_root,
        split="trainval",
        target_types="segmentation",
        transforms=PetTransform(),
        download=True,
    )
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator.manual_seed(seed)
    )
    return train_dataset,val_dataset
