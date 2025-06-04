import timm
import torch.nn as nn

from utils.model_utils import get_device


def get_dino_backbone_model(num_classes: int = 100):
    device = get_device()
    backbone = timm.create_model(
        "vit_small_patch16_224_dino", pretrained=True
    ) 

    for param in backbone.parameters():
        param.requires_grad = False

    backbone.head = nn.Linear(384, num_classes)
    return backbone.to(device)
