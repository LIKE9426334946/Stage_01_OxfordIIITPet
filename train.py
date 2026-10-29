import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from utils.config import load_config
from utils.dataset import load_dataset
from utils.metrics import BinarySegmentationMetrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, criterion, device, threshold, description, optimizer=None):
    """传入 optimizer 时训练；不传时执行验证，不计算梯度。"""
    training = optimizer is not None
    model.train(training)
    metrics = BinarySegmentationMetrics(device, threshold)
    loss_sum = 0.0
    sample_count = 0
    progress = tqdm(loader, desc=description, dynamic_ncols=True)

    with torch.set_grad_enabled(training):
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            # logits 和 masks 都是 [B, 1, H, W]。
            # BCEWithLogitsLoss 内部包含 sigmoid，因此这里直接传入 logits。
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
            metrics.update(logits, masks)
            progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}")

    return {"loss": loss_sum / sample_count, **metrics.compute()}


def format_scores(scores):
    return (
        f"Loss={scores['loss']:.4f} | IoU={scores['iou']:.4f} | "
        f"mIoU={scores['miou']:.4f} | Dice={scores['dice']:.4f} | "
        f"Acc={scores['accuracy']:.4f} | Precision={scores['precision']:.4f} | "
        f"Recall={scores['recall']:.4f}"
    )


def main(argv=None):
    config = load_config(argv)
    training = config["training"]
    set_seed(training["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = config.get("metrics", {}).get("threshold", 0.5)
    print("使用的设备为：", device)
    print("使用的模型为：", config["model"]["name"])

    train_dataset, val_dataset = load_dataset(config)
    loader_options = {
        "batch_size": training["batch_size"],
        "num_workers": training.get("num_workers", 0),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(training["seed"]),
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    print("训练集数量：", len(train_dataset))
    print("验证集数量：", len(val_dataset))

    model = build_model(config["model"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])

    # 每次训练使用独立目录，防止覆盖之前的权重和日志。
    run_name = f"{config['model']['name']}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output_dir = Path(training["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    print("本次训练结果保存到：", output_dir)

    metric_names = ("loss", "iou", "miou", "dice", "accuracy", "precision", "recall")
    fieldnames = ["epoch"] + [
        f"{phase}_{name}" for phase in ("train", "val") for name in metric_names
    ]
    best_miou = -1.0
    best_epoch = 0

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, training["epochs"] + 1):
            train_scores = run_epoch(
                model, train_loader, criterion, device, threshold,
                f"Epoch {epoch}/{training['epochs']} Train", optimizer=optimizer,
            )
            val_scores = run_epoch(
                model, val_loader, criterion, device, threshold,
                f"Epoch {epoch}/{training['epochs']} Val",
            )
            print("训练：", format_scores(train_scores))
            print("验证：", format_scores(val_scores))

            row = {"epoch": epoch}
            row.update({f"train_{key}": value for key, value in train_scores.items()})
            row.update({f"val_{key}": value for key, value in val_scores.items()})
            writer.writerow(row)
            file.flush()

            # 保存纯模型 state_dict，后续直接用 model.load_state_dict 加载。
            torch.save(model.state_dict(), output_dir / "last.pth")
            if val_scores["miou"] > best_miou:
                best_miou = val_scores["miou"]
                best_epoch = epoch
                torch.save(model.state_dict(), output_dir / "best.pth")
                print(f"已保存最佳模型：epoch={epoch}, val_mIoU={best_miou:.4f}")

    print(f"训练结束。最佳 epoch={best_epoch}，验证集 mIoU={best_miou:.4f}")
    print("最佳模型：", output_dir / "best.pth")


if __name__ == "__main__":
    main()
