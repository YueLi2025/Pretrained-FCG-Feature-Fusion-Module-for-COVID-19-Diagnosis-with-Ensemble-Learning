import torch.nn as nn
from torchvision import models
from typing import Union


class MyModel(nn.Module):
    """Custom neural network model that supports multiple backbone architectures.

    Args:
        n_out (int): Number of output classes.
        backbone (str): Name of the backbone model. Supported options:
            - 'resnet34'
            - 'resnet50'
            - 'densenet121'
            - 'vgg19_bn'
            - 'vgg11'
    """

    def __init__(self, n_out: int, backbone: str):
        super(MyModel, self).__init__()

        self.backbone = self._initialize_backbone(backbone, n_out)

    def _initialize_backbone(self, backbone: str, n_out: int) -> Union[nn.Module, None]:
        """Initializes the backbone network with pre-trained weights and modifies the final layer.

        Args:
            backbone (str): The chosen backbone model.
            n_out (int): Number of output classes.

        Returns:
            nn.Module: The modified backbone model.
        """
        if backbone == "resnet34":
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(512, n_out)

        elif backbone == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, n_out)

        elif backbone == "densenet121":
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(1024, n_out)

        elif backbone == "vgg19_bn":
            model = models.vgg19_bn(pretrained=True)
            model.classifier[-1] = nn.Linear(4096, n_out)

        elif backbone == "vgg11":
            model = models.vgg11_bn(pretrained=True)
            model.classifier[-1] = nn.Linear(4096, n_out)

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from ['resnet34', 'resnet50', 'densenet121', 'vgg19_bn', 'vgg11'].")

        return model

    def forward(self, x):
        """Forward pass of the model."""
        return self.backbone(x)
