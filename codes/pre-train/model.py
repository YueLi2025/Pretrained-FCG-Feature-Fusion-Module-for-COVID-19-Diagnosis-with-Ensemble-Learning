import torch.nn as nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, n_out, backbone):
        super(MyModel, self).__init__()
        if backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
            self.backbone.fc = nn.Linear(512, n_out)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Linear(2048, n_out)
        elif backbone == "densenet121":
            self.backbone = models.densenet121(pretrained=True)
            self.backbone.classifier = nn.Linear(1024, n_out, bias=True)
        elif backbone == "vgg19_bn":
            self.backbone = models.vgg19_bn(pretrained=True)
            self.backbone.classifier[-1] = nn.Linear(4096, n_out)
        elif backbone == "vgg11":
            self.backbone = models.vgg11_bn(pretrained=True)
            self.backbone.classifier[-1] = nn.Linear(4096, n_out)

    def forward(self, x):
        x = self.backbone(x)
        return x
