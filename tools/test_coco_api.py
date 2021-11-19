import sys
import os
import json

from pycocotools.coco import COCO

ann_file = ".json"
coco = COCO(annotation_file=ann_file)

print("coco\nimage size [%05d]\t annotation size [%05d]\t category size [%05d]\ndone" % (len(coco.imgs), len(coco.anns), len(coco.cats)))