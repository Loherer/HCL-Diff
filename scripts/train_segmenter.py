from __future__ import annotations

import argparse

from hcl_diff.config import load_config
from hcl_diff.training.segmenter import train_segmenter


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 2.5D lesion segmentation model.")
    parser.add_argument("--config", required=True, help="Segmentation YAML configuration")
    arguments = parser.parse_args()
    checkpoint = train_segmenter(load_config(arguments.config))
    print(checkpoint)


if __name__ == "__main__":
    main()
