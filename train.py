import torch


from torch.utils.data import DataLoader

from utils.config import load_config
from utils.dataset import load_dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用的设备为：", device)

    config = load_config()
    model_name = config["model"]["name"]
    print("使用的模型为：", model_name)

    torch.manual_seed(config["training"]["seed"])

    batch_size = config["training"]["batch_size"]
    train_dataset, val_dataset = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)


if __name__ == "__main__":
    main()
