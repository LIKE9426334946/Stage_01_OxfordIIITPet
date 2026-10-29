import argparse
from pathlib import Path

import yaml


def load_config(argv=None):
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="加载自定义参数")
    parser.add_argument(
        "--config", type=Path, default=project_root / "config.yaml", help="配置文件路径"
    )
    parser.add_argument("--root", type=str, default=None, help="覆盖当前数据集根目录")
    parser.add_argument("--model", type=str, default=None, help="覆盖当前模型名称")
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # YAML 中的相对路径统一以该配置文件所在目录为起点。
    for section, key in (("dataset", "root"), ("training", "output_dir")):
        path = Path(config[section][key]).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        config[section][key] = str(path.resolve())

    if args.root is not None:
        # 命令行 --root 的相对路径以启动命令的目录为起点。
        config["dataset"]["root"] = str(Path(args.root).expanduser().resolve())
    if args.model is not None:
        config["model"]["name"] = args.model

    return config


if __name__ == "__main__":
    print(load_config())
