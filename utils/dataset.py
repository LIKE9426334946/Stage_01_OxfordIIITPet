import numpy as np
import torch

from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torch.utils.data import random_split


class PetTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image, mask):
        image = F.resize(
            image, self.image_size, interpolation=InterpolationMode.BILINEAR
        )
        mask = F.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)
        image = F.to_tensor(image)

        mask = np.array(mask)
        # 原标签 1=宠物，2=背景，3=不确定边界；这里将 1 和 3 都并入前景。
        mask = (mask != 2).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0) # 变为[1,256,256]

        return image, mask


def load_dataset(config):
    image_size = config["dataset"]["image_size"]
    image_root = config["dataset"]["root"]
    val_ratio = config["dataset"]["val_ratio"]
    seed = config["training"]["seed"]

    dataset = OxfordIIITPet(
        root=image_root,
        split="trainval",
        target_types="segmentation",
        transforms=PetTransform(image_size),
        download=True,
    )
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("训练集和验证集都必须至少包含一个样本，请检查 val_ratio。")
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    return train_dataset, val_dataset
