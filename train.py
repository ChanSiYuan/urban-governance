import os
import os.path as osp
import sys
import glob
import xml.etree.ElementTree as ElementTree
import argparse


from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators

# label change
# label_map = {
#     "fixed": 0,
#     "mobile": 1,
#     "bag": 2

    # "leaf": 0,
    # "paper": 1
# }

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


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")

    # Basic setting
    parser.add_argument('-t', '--task', default="", help="Choose the label for current task [ctrashc, strashc, "
                                                          "truck, trash, flotage, blot]")
    parser.add_argument('-o', '--result_output', default="/home/csy/research/cs_zz/can/ctrashc", help='Set the result '
                                                                                                 'store folder')
    parser.add_argument('-d', '--data_dir', default='/home/csy/data/cs_zz/can/ctrashc', help='Set the data folder '
                                                                                              'path')
    parser.add_argument('-g', '--gpuid', default='2', help='Set the available gpu')

    return parser


def get_dicts(args, mode='train'):
    train_dir = osp.join(args.data_dir, mode + '.txt')
    # print(train_dir)
    with open(train_dir, 'r') as f:
        xml_index_markers = f.readlines()

    print(f'xml file len: {len(xml_index_markers)}')

    dataset_dicts = []
    for idx, xml_index in enumerate(xml_index_markers):
        xml_file = osp.join(args.data_dir, 'xmls', xml_index.strip() + '.xml')
        if not osp.isfile(xml_file):
            print(f'Path {xml_file} not a file')
            continue

        tree = ElementTree.parse(xml_file)
        root = tree.getroot()

        filename = osp.join(args.data_dir, 'images', xml_index.strip() + '.jpg')
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


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # dataset change
    cfg.DATASETS.TRAIN = ("cur_train",)
    cfg.DATASETS.TEST = ("cur_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_zoo[args.task])
    cfg.OUTPUT_DIR = args.result_output
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.freeze()

    return cfg


if __name__ == '__main__':

    args = get_parser().parse_args()
    print('\n--- Current args ---\n')
    print(args)

    assert args.task in label_zoo, "Please check and set the task"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    # dataset_name_change
    for mode in ['train', 'val']:
        DatasetCatalog.register('cur_' + mode, lambda mode=mode: get_dicts(args, mode))
        MetadataCatalog.get('cur_' + mode).set(thing_classes=list(label_zoo[args.task].keys()))

    cfg = setup_cfg(args)

    print('\n--- Current configs ---\n')
    print(cfg)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation with AP metric

    # dataset change
    evaluator = COCOEvaluator("cur_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "cur_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)

