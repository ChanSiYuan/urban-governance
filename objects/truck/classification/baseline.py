import torch
import torch.nn as nn

from torchvision.models import resnet50, resnet101

from .backbone import se_resnext101_32x4d, senet154
from .BasicModule import BasicModule

FACTORY = {
    'ResNet50': resnet50,
    'ResNet101': resnet101,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'senet154': senet154
}


class ResModelWide(BasicModule):
    def __init__(self, num_classes=9, arch='ResNet50', stride=1, pretrained=True):
        assert arch in FACTORY
        super().__init__()

        self.num_classes = num_classes
        self.arch = arch
        self.stride = stride

        # BaseNet
        if 'ResNet' in self.arch:
            resnet = FACTORY[self.arch](pretrained=pretrained)
            if stride == 1:
                resnet.layer4[0].downsample[0].stride = (1, 1)
                resnet.layer4[0].conv2.stride = (1, 1)

            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,  # res_conv2
                resnet.layer2,  # res_conv3
                resnet.layer3,  # res_conv4
                resnet.layer4
            )
        elif 'se' in self.arch:
            se_resnext = FACTORY[self.arch](pretrained='imagenet')
            self.backbone = nn.Sequential(
                se_resnext.layer0,
                se_resnext.layer1,  # res_conv2
                se_resnext.layer2,  # res_conv3
                se_resnext.layer3,  # res_conv4
                se_resnext.layer4
            )
        else:
            raise KeyError("Unknown Arch: ", self.arch)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.bottleneck = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5)
        )
        self.classifier = nn.Linear(512, self.num_classes)

        for m in self.bottleneck:
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.xavier_uniform_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self, x):
        x = self.backbone(x)
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat((x_avg, x_max), dim=1)
        x = x.reshape(x.size(0), -1)
        x = self.bottleneck(x)
        cls = self.classifier(x)

        return cls


class ResModel(BasicModule):
    def __init__(self, num_classes=9, arch='ResNet50', stride=1, pretrained=True):
        assert arch in FACTORY

        super().__init__()
        self.num_classes = num_classes
        self.arch = arch
        self.stride = stride

        # BaseNet
        if 'ResNet' in self.arch:
            resnet = FACTORY[self.arch](pretrained=pretrained)
            if stride == 1:
                resnet.layer4[0].downsample[0].stride = (1, 1)
                resnet.layer4[0].conv2.stride = (1, 1)

            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,  # res_conv2
                resnet.layer2,  # res_conv3
                resnet.layer3,  # res_conv4
                resnet.layer4
            )
        elif 'se' in self.arch:
            se_resnext = FACTORY[self.arch](pretrained='imagenet')
            self.backbone = nn.Sequential(
                se_resnext.layer0,
                se_resnext.layer1,  # res_conv2
                se_resnext.layer2,  # res_conv3
                se_resnext.layer3,  # res_conv4
                se_resnext.layer4
            )
        else:
            raise KeyError("Unknown Arch: ", self.arch)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5)
        )
        self.classifier = nn.Linear(512, self.num_classes)

        for m in self.bottleneck:
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.xavier_uniform_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = x.reshape(x.size(0), -1)
        x = self.bottleneck(x)
        cls = self.classifier(x)

        return cls
