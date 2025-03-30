"""
MaxViT: Multi-Axis Vision Transformer

https://arxiv.org/abs/2204.01697
Code is extended from torchvision.models.maxvit.py
"""	

from typing import Any, List, Optional

from torchvision.models._api import WeightsEnum, register_model
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models.maxvit import MaxVit, MaxVit_T_Weights

__all__ = ["maxvit_s"]


def _maxvit(
    # stem parameters
    stem_channels: int,
    # block parameters
    block_channels: List[int],
    block_layers: List[int],
    stochastic_depth_prob: float,
    # partitioning parameters
    partition_size: int,
    # transformer parameters
    head_dim: int,
    # Weights API
    weights: Optional[WeightsEnum] = None,
    progress: bool = False,
    # kwargs,
    **kwargs: Any,
) -> MaxVit:

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes",
                              len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "input_size", weights.meta["min_size"])

    input_size = kwargs.pop("input_size", (224, 224))

    model = MaxVit(
        stem_channels=stem_channels,
        block_channels=block_channels,
        block_layers=block_layers,
        stochastic_depth_prob=stochastic_depth_prob,
        head_dim=head_dim,
        partition_size=partition_size,
        input_size=input_size,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(
            progress=progress, check_hash=True))

    return model


@register_model()
@handle_legacy_interface(weights=("pretrained", MaxVit_T_Weights.IMAGENET1K_V1))
def maxvit_s(*, weights: Optional[MaxVit_T_Weights] = None, progress: bool = True, **kwargs: Any) -> MaxVit:
    """
    Constructs a maxvit_t architecture from
    `MaxViT: Multi-Axis Vision Transformer <https://arxiv.org/abs/2204.01697>`_.

    Args:
        weights (:class:`~torchvision.models.MaxVit_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MaxVit_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.maxvit.MaxVit``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/maxvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MaxVit_T_Weights
        :members:
    """
    weights = MaxVit_T_Weights.verify(weights)

    return _maxvit(
        stem_channels=64,
        block_channels=[96, 192, 384, 768],
        block_layers=[2, 2, 5, 2],
        head_dim=32,
        stochastic_depth_prob=0.2,
        partition_size=7,
        weights=weights,
        progress=progress,
        **kwargs,
    )
