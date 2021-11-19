import os
import argparse
import shutil
import logging


def get_parser():
    parser = argparse.ArgumentParser(description="frame rename script")
    parser.add_argument("--src_path", default="", help="path that source frames need to rename")
    parser.add_argument("--dst_path", default="", help="path that target frames generated")
    parser.add_argument("--name_width", default="4", help="name width of the new rename frame, [ 4 | 5 | 6 ]")
    parser.add_argument("--file_ext", default="jpg", help="extension name of the file, [ jpg | xml ]")
    parser.add_argument("--start_cnt", default="0", help="rename file start count")
    parser.add_argument("--duplicate", default=True, help="rename file and whether duplicate this file")
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


def check_folder(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def main():
    logger = get_logger()
    args = get_parser().parse_args()
    logger.info(f"args: {args}")
    dst_path = args.dst_path
    if not args.dst_path:
        dst_path = os.path.join(args.src_path, "..", "rename")
    duplicate_path = os.path.join(args.src_path, "..", f"backup_{args.file_ext}")
    check_folder([dst_path, duplicate_path])

    files = os.listdir(args.src_path)
    files.sort()
    start_count = int(args.start_cnt)
    # TODO: add folder check func
    # TODO: add check file-ext func
    # TODO: add recursive collect frame
    process_count = start_count
    for frame_name in files:
        frame_base_name, file_ext = frame_name.split(".")[0], frame_name.split(".")[1]
        if file_ext != args.file_ext:
            continue

        process_count += 1
        old_name = os.path.join(args.src_path, frame_name)

        if args.name_width == "4":
            # total no sorted, sort by ourselves
            # new_name = os.path.join(args.dst_path, f"{process_count:04}.{args.file_ext}")
            frame_name_split = frame_base_name.split("-")
            padding_name = frame_name_split[1].rjust(4, "0")
            new_name = os.path.join(dst_path, f"{frame_name_split[0]}-{padding_name}.{args.file_ext}")
        elif args.name_width == "5":
            new_name = os.path.join(dst_path, f"{process_count:05}.{args.file_ext}")
        elif args.name_width == "6":
            new_name = os.path.join(dst_path, f"{process_count:06}.{args.file_ext}")
        else:
            raise ValueError("parameter [name width] value error")
        if args.duplicate:
            duplicate_name = os.path.join(duplicate_path, f"{frame_base_name}.{args.file_ext}")
            shutil.copy(src=old_name, dst=duplicate_name)
            logger.info(f"copy frame: {duplicate_name}")

        os.rename(src=old_name, dst=new_name)
        logger.info(f"src name: {old_name} --> dst name: {new_name}")
    logger.info(f"rename complete. start count: {start_count}, end count: {process_count}, total: {process_count - start_count}")


if __name__ == "__main__":
    main()
