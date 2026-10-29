import argparse
import yaml
from pathlib import Path


def load_config():
    parser = argparse.ArgumentParser(description="加载自定义参数")
    parser.add_argument("--root", type=str, default=None, help="覆盖当前数据集根目录")
    parser.add_argument("--model", type=str, default=None, help="覆盖当前模型名称")
    args = parser.parse_args()

    with open("./config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.root is not None:
        config["dataset"]["root"] = args.root
    if args.model is not None:
        config["model"]["name"] = args.model

    return config


if __name__ == "__main__":
    config = load_config()
    print(config)

