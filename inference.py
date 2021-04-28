import os
import os.path as osp
import glob
import sys
import xml.etree.ElementTree as ElementTree
import argparse

sys.path.extend(['..', '../..'])

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from shutil import copyfile
from lxml import etree
import time

label_zoo = dict({
    "strashc": dict({
        "fixed": 0,
        "mobile": 1,
        "bag": 2
    }),
    "ctrashc": dict({
        "mobile": 0,
        "bag": 1
    }),
    "trash": dict({
        "leaf": 0,
        "paper": 1
    }),
    "truck": dict({

    }),
    "flotage": dict({

    }),
    "blot": dict({

    })
})

data_dir = '/home/csy/data/cszz/can/ctrashc'


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")

    # Basic setting
    parser.add_argument('-t', '--task', default="", help="Choose the label for current task [ctrashc, strashc, "
                                                         "truck, trash, flotage, blot]")

    return parser


def get_dicts(data_dir, mode='train'):
    train_dir = osp.join(data_dir, mode + '.txt')
    # print(train_dir)
    with open(train_dir, 'r') as f:
        xml_index_markers = f.readlines()

    print(f'xml file len: {len(xml_index_markers)}')

    dataset_dicts = []
    for idx, xml_index in enumerate(xml_index_markers):
        xml_file = osp.join(data_dir, 'xmls', xml_index.strip() + '.xml')
        if not osp.isfile(xml_file):
            print(f'Path {xml_file} not a file')
            continue

        tree = ElementTree.parse(xml_file)
        root = tree.getroot()

        filename = osp.join(data_dir, 'images', xml_index.strip() + '.jpg')
        if not osp.isfile(filename):
            print(f'Path {filename} not a file')
            continue

        record = dict({
            'file_name': filename,
            'image_id': idx,
            'height': int(root.find('size').find('height').text),
            'width': int(root.find('size').find('width').text)
        })
        # print(record)

        objs = []
        for member in root.findall('object'):
            if len(member) < 5:
                print(f'Member length: {len(member)}')
                continue

            if label_zoo[args.task].get(member.find('name').text) is None:
                print('Error label')
                continue

            obj = dict({
                'bbox': [
                    int(member.find('bndbox')[0].text),
                    int(member.find('bndbox')[1].text),
                    int(member.find('bndbox')[2].text),
                    int(member.find('bndbox')[3].text)
                ],
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': [],
                'category_id': label_zoo[args.task][member.find('name').text],
                'iscrowd': 0
            })
            objs.append(obj)

        record['annotations'] = objs
        # print(record)
        dataset_dicts.append(record)

    return dataset_dicts


class Gen_Annotation:
    def __init__(self, filename):
        self.root = etree.Element("annotation")
        file = etree.SubElement(self.root, "filename")
        file.text = filename

    def set_size(self, width, height, channel):
        source = etree.SubElement(self.root, "source")
        database = etree.SubElement(source, "database")
        database.text = "ctrashc"

        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(width)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "channel")
        channeln.text = str(channel)

        segmented = etree.SubElement(self.root, "segmented")
        segmented.text = '0'

    def savefile(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding="utf-8")

    def add_pic_attr(self, label, x1, y1, x2, y2):
        obj = etree.SubElement(self.root, "object")
        namen = etree.SubElement(obj, "name")
        namen.text = label

        pose = etree.SubElement(obj, "pose")
        pose.text = "Unspecified"

        bndbox = etree.SubElement(obj, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(x1)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(y1)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(x2)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(y2)


if __name__ == "__main__":

    args = get_parser()
    img_path = r"/home/csy/data/cszz/test/test_data"

    # choose the output folder
    output_path = r"/home/csy/data/cszz/test/test_result"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(osp.join(output_path, "images"), exist_ok=True)
    os.makedirs(osp.join(output_path, "xmls"), exist_ok=True)
    os.makedirs(osp.join(output_path, "images_with_labels"), exist_ok=True)

    # choose the cuda number
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    for mode in ['train', 'val']:
        DatasetCatalog.register('cur_' + mode, lambda mode=mode: get_dicts(data_dir, mode))
        MetadataCatalog.get('cur_' + mode).set(thing_classes=list(label_zoo[args.task].keys()))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("cur_train",)
    cfg.DATASETS.TEST = ("cur_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 3000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_zoo[args.task])

    # choose the model weight
    # cfg.MODEL.WEIGHTS = os.path.join("/data/csy/cs_zz/final_models", "model_final_ctrashcan.pth")
    cfg.MODEL.WEIGHTS = osp.join("/home/csy/project/cszz/checkpoints/weights", "ctrashc_v20.pth")

    predictor = DefaultPredictor(cfg)

    cnt = 0
    for img_name in os.listdir(img_path):
        cnt += 1
        if img_name.startswith("."):
            continue
        im = cv2.imread(os.path.join(img_path, img_name))
        start_time = time.time()
        outputs = predictor(im)
        end_time = time.time()

        print(f"ID: {cnt} || image name: {img_name} || cost time: {end_time - start_time:2f} s")
        print(f"result: {outputs}\n")

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(os.path.join(output_path, "images_with_labels", img_name.split(".")[0] + ".jpg"),
            v.get_image()[:, :, ::-1])

        res = outputs["instances"]._fields["pred_boxes"]
        output_size = len(res)
        # if output_size == 0:
        #     continue

        h, w, c = im.shape

        output = outputs["instances"].to("cpu")
        boxes = output._fields["pred_boxes"].tensor.tolist()
        scores = output._fields["scores"].tolist()
        classes = output._fields["pred_classes"].tolist()

        anno = Gen_Annotation(osp.join(output_path, "images", img_name.split(".")[0]))
        anno.set_size(w, h, c)

        label_keys = list(label_zoo[args.task].keys())
        for i in range(len(classes)):
            anno.add_pic_attr(
                label_keys[classes[i]],
                int(boxes[i][0]),
                int(boxes[i][1]),
                int(boxes[i][2]),
                int(boxes[i][3])
            )

        anno.savefile(osp.join(output_path, "xmls", img_name.split(".")[0]) + ".xml")
        copyfile(os.path.join(img_path, img_name), osp.join(output_path, "images", img_name.split(".")[0]) + ".jpg")
