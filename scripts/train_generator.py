from __future__ import annotations

import argparse

from hcl_diff.config import load_config
from hcl_diff.training.generator import train_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a C0, C1 or C2 lesion generator.")
    parser.add_argument("--config", required=True, help="Generator YAML configuration")
    arguments = parser.parse_args()
    checkpoint = train_generator(load_config(arguments.config))
    print(checkpoint)


if __name__ == "__main__":
    main()
