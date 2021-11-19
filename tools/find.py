import sys
import os
from collections import defaultdict

def main():
    files = os.listdir(r"Y:\cs-urban_goverance\data\ctrashc-voc\VOCdevkit\VOC2017\backup")
    record = defaultdict(int)
    for file in files:
        file_base_name = file.split(".")[0]
        # print(file_base_name)
        record[file_base_name] += 1
    # print(record)
    for key, value in record.items():
        if value == 1:
            print(key)

if __name__ == "__main__":
    main()