import torch.nn as nn

from .baseline import ResModel, ResModelWide
from .losses import CrossEntropyLabelSmooth, FocalLoss
from .default import _C as cfg

MODEL_FACTORY = {
    'ResModel': ResModel,
    'ResModelWide': ResModelWide,
}


def make_loss(cfg, num_classes):
    if cfg.SOLVER.LOSS == 'softmax':
        return nn.CrossEntropyLoss()
    elif cfg.SOLVER.LOSS == 'softmax_LS':
        return CrossEntropyLabelSmooth(num_classes=num_classes)
    elif cfg.SOLVER.LOSS == 'Focal_Loss':
        return FocalLoss()
    else:
        raise KeyError("Unknown loss: ", cfg.SOLVER.LOSS)


def make_model(cfg, num_classes):
    assert cfg.MODEL.NAME in MODEL_FACTORY

    model = MODEL_FACTORY[cfg.MODEL.NAME](num_classes=num_classes, arch=cfg.MODEL.ARCH, stride=cfg.MODEL.STRIDE)

    return model
