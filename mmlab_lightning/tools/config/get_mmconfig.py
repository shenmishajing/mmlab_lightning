import argparse

from mmengine.hub import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="Get config file from mmlab repos")
    parser.add_argument("config", help="config file")
    parser.add_argument(
        "--save-path", help="path to save config", default="config.yaml"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    get_config(args.config, True).dump(args.save_path)


if __name__ == "__main__":
    main()
