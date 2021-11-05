import os
import argparse
import shutil
import logging


def get_parser():
    parser = argparse.ArgumentParser(description="frame rename script")
    parser.add_argument("--src_path", default="", help="path that source frames need to rename")
    parser.add_argument("--dst_path", default="", help="path that target frames generated")
    return parser


def get_logger():
    logger = logging.getLogger("rename script logger")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main():
    logger = get_logger()

    args = get_parser().parse_args()
    frames = os.listdir(args.src_path)
    name_count = 0
    for frame_name in frames:
        name_count += 1
        frame_base_name = frame_name.split(".")[0]
        old_name = os.path.join(args.src_path, frame_name)
        new_name = os.path.join(args.dst_path, f"{name_count:04}.jpg")
        duplicate_name = os.path.join(args.src_path, f"{frame_base_name}_backup.jpg")
        shutil.copy(src=old_name, dst=duplicate_name)
        logger.info(f"copy frame: {duplicate_name}")

        os.rename(src=old_name, dst=new_name)
        logger.info(f"src name: {old_name} --> dst name: {new_name}")
    logger.info("rename complete")


if __name__ == "__main__":
    main()
