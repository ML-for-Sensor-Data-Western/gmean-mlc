from multilabel_models.tresnet_model import TResNet
from multilabel_models.gcn_models import ResNet_GCNN, ResNet_KSSNET
from multilabel_models.create_adjacency_matrix import normalize_adjacency_matrix
import torchvision.models as torch_models

import torch
import numpy as np


def tresnet_m(num_classes, pretrained = False, mtl_heads=False, **kwargs):
    """ Constructs a medium TResnet model.
    """
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=3,
                    remove_aa_jit=True, mtl_heads=mtl_heads)
    
    if pretrained:
        # Load state dicts!
        pass

    return model


def tresnet_l(num_classes, pretrained = False, mtl_heads=False, **kwargs):
    """ Constructs a large TResnet model.
    """
    model = TResNet(layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=3, width_factor=1.2,
                    remove_aa_jit=True, mtl_heads=mtl_heads)
    
    if pretrained:
        # Load state dicts!
        pass

    return model


def tresnet_xl(num_classes, pretrained = False, mtl_heads=False, **kwargs):
    """ Constructs an extra-large TResnet model.
    """
    model = TResNet(layers=[4, 5, 24, 3], num_classes=num_classes, in_chans=3, width_factor=1.3,
                    remove_aa_jit=True, mtl_heads=mtl_heads)
    
    if pretrained:
        # Load state dicts!
        pass

    return model