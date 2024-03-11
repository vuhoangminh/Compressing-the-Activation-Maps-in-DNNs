""" PyTorch EfficientDet model

Based on official Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
Paper: https://arxiv.org/abs/1911.09070

Hacked together by Ross Wightman
"""

import logging
import math
from collections import OrderedDict
from functools import partial
from typing import List, Callable, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from timm import create_model

try:
    from timm.layers import create_conv2d, create_pool2d, get_act_layer
except ImportError:
    from timm.models.layers import create_conv2d, create_pool2d, get_act_layer

import itertools

from omegaconf import OmegaConf

from engine.compression import (
    WaveletTransformCompression,
    DCTCompression,
    RandomCompression,
    ConditionalContext,
)

from WaveletCompressedConvolution.WCC.transform_model import (
    wavelet_module as WCC_wavelet,
)


_DEBUG = False
_USE_SCALE = False
_ACT_LAYER = get_act_layer("silu")


""" RetinaNet / EfficientDet Anchor Gen

Adapted for PyTorch from Tensorflow impl at
    https://github.com/google/automl/blob/6f6694cec1a48cdb33d5d1551a2d5db8ad227798/efficientdet/anchors.py

Hacked together by Ross Wightman, original copyright below
"""
# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Anchor definition.

This module is borrowed from TPU RetinaNet implementation:
https://github.com/tensorflow/tpu/blob/master/models/official/retinanet/anchors.py
"""
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms, remove_small_boxes


# The minimum score to consider a logit for identifying detections.
MIN_CLASS_SCORE = -5.0

# The score for a dummy detection
_DUMMY_DETECTION_SCORE = -1e5


"""EfficientDet Configurations

Adapted from official impl at https://github.com/google/automl/tree/master/efficientdet

TODO use a different config system (OmegaConfig -> Hydra?), separate model from train specific hparams
"""

from omegaconf import OmegaConf
from copy import deepcopy


def default_detection_model_configs():
    """Returns a default detection configs."""
    h = OmegaConf.create()

    # model name.
    h.name = "tf_efficientdet_d1"

    h.backbone_name = "tf_efficientnet_b1"
    h.backbone_args = None  # FIXME sort out kwargs vs config for backbone creation
    h.backbone_indices = None

    # model specific, input preprocessing parameters
    h.image_size = (640, 640)

    # dataset specific head parameters
    h.num_classes = 90

    # feature + anchor config
    h.min_level = 3
    h.max_level = 7
    h.num_levels = h.max_level - h.min_level + 1
    h.num_scales = 3
    h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    # ratio w/h: 2.0 means w=1.4, h=0.7. Can be computed with k-mean per dataset.
    # aspect ratios can be specified as below too, pairs will be calc as sqrt(val), 1/sqrt(val)
    # h.aspect_ratios = [1.0, 2.0, 0.5]
    h.anchor_scale = 4.0

    # FPN and head config
    h.pad_type = (
        "same"  # original TF models require an equivalent of Tensorflow 'SAME' padding
    )
    h.act_type = "swish"
    h.norm_layer = None  # defaults to batch norm when None
    h.norm_kwargs = dict(eps=0.001, momentum=0.01)
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_channels = 88
    h.separable_conv = True
    h.apply_resample_bn = True
    h.conv_bn_relu_pattern = False
    h.downsample_type = "max"
    h.upsample_type = "nearest"
    h.redundant_bias = (
        True  # original TF models have back to back bias + BN layers, not necessary!
    )
    h.head_bn_level_first = False  # change order of BN in head repeat list of lists, True for torchscript compat
    h.head_act_type = None  # activation for heads, same as act_type if None

    h.fpn_name = None
    h.fpn_config = None
    h.fpn_drop_path_rate = 0.0  # No stochastic depth in default. NOTE not currently used, unstable training

    # classification loss (used by train bench)
    h.alpha = 0.25
    h.gamma = 1.5
    h.label_smoothing = (
        0.0  # only supported if legacy_focal == False, haven't produced great results
    )
    h.legacy_focal = (
        False  # use legacy focal loss (less stable, lower memory use in some cases)
    )
    h.jit_loss = False  # torchscript jit for loss fn speed improvement, can impact stability and/or increase mem usage

    # localization loss (used by train bench)
    h.delta = 0.1
    h.box_loss_weight = 50.0

    # nms
    h.soft_nms = False  # use soft-nms, this is incredibly slow
    h.max_detection_points = 5000  # max detections for post process, input to NMS
    h.max_det_per_image = 100  # max detections per image limit, output of NMS

    return h


tf_efficientdet_lite_common = dict(
    fpn_name="bifpn_sum",
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    act_type="relu6",
)


efficientdet_model_param_dict = dict(
    # Models with PyTorch friendly padding and my PyTorch pretrained backbones, training TBD
    efficientdet_d0=dict(
        name="efficientdet_d0",
        backbone_name="efficientnet_b0",
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.1),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d0-f3276ba8.pth",
    ),
    efficientdet_d1=dict(
        name="efficientdet_d1",
        backbone_name="efficientnet_b1",
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type="",
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d1-bb7e98fe.pth",
    ),
    efficientdet_d2=dict(
        name="efficientdet_d2",
        backbone_name="efficientnet_b2",
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        pad_type="",
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url="",  # no pretrained weights yet
    ),
    efficientdet_d3=dict(
        name="efficientdet_d3",
        backbone_name="efficientnet_b3",
        image_size=(896, 896),
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        pad_type="",
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url="",  # no pretrained weights yet
    ),
    efficientdet_d4=dict(
        name="efficientdet_d4",
        backbone_name="efficientnet_b4",
        image_size=(1024, 1024),
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url="",  # no pretrained weights yet
    ),
    efficientdet_d5=dict(
        name="efficientdet_d5",
        backbone_name="efficientnet_b5",
        image_size=(1280, 1280),
        fpn_channels=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url="",  # no pretrained weights yet
    ),
    efficientdetv2_dt=dict(
        name="efficientdetv2_dt",
        backbone_name="efficientnetv2_rw_t",
        image_size=(768, 768),
        fpn_channels=128,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        aspect_ratios=[1.0, 2.0, 0.5],
        pad_type="",
        downsample_type="bilinear",
        upsample_type="bilinear",
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdetv2_rw_dt_agc-ad8b8c96.pth",
    ),
    efficientdetv2_ds=dict(
        name="efficientdetv2_ds",
        backbone_name="efficientnetv2_rw_s",
        image_size=(1024, 1024),
        fpn_channels=256,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        aspect_ratios=[1.0, 2.0, 0.5],
        pad_type="",
        downsample_type="bilinear",
        upsample_type="bilinear",
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientnetv2_rw_ds_agc-cf589293.pth",
    ),
    # My own experimental configs with alternate models, training TBD
    # Note: any 'timm' model in the EfficientDet family can be used as a backbone here.
    resdet50=dict(
        name="resdet50",
        backbone_name="resnet50",
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type="",
        act_type="relu",
        redundant_bias=False,
        separable_conv=False,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/resdet50_416-08676892.pth",
    ),
    cspresdet50=dict(
        name="cspresdet50",
        backbone_name="cspresnet50",
        image_size=(768, 768),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type="",
        act_type="leaky_relu",
        head_act_type="silu",
        downsample_type="bilinear",
        upsample_type="bilinear",
        redundant_bias=False,
        separable_conv=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/cspresdet50b-386da277.pth",
    ),
    cspresdext50=dict(
        name="cspresdext50",
        backbone_name="cspresnext50",
        image_size=(640, 640),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type="",
        act_type="leaky_relu",
        redundant_bias=False,
        separable_conv=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url="",
    ),
    cspresdext50pan=dict(
        name="cspresdext50pan",
        backbone_name="cspresnext50",
        image_size=(640, 640),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=88,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        act_type="leaky_relu",
        fpn_name="pan_fa",  # PAN FPN experiment
        redundant_bias=False,
        separable_conv=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/cspresdext50pan-92fdd094.pth",
    ),
    cspdarkdet53=dict(
        name="cspdarkdet53",
        backbone_name="cspdarknet53",
        image_size=(640, 640),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type="",
        act_type="leaky_relu",
        redundant_bias=False,
        separable_conv=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        backbone_indices=(3, 4, 5),
        url="",
    ),
    cspdarkdet53m=dict(
        name="cspdarkdet53m",
        backbone_name="cspdarknet53",
        image_size=(768, 768),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=96,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type="",
        fpn_name="qufpn_fa",
        act_type="leaky_relu",
        head_act_type="mish",
        downsample_type="bilinear",
        upsample_type="bilinear",
        redundant_bias=False,
        separable_conv=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        backbone_indices=(3, 4, 5),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/cspdarkdet53m-79062b2d.pth",
    ),
    mixdet_m=dict(
        name="mixdet_m",
        backbone_name="mixnet_m",
        image_size=(512, 512),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.1),
        url="",  # no pretrained weights yet
    ),
    mixdet_l=dict(
        name="mixdet_l",
        backbone_name="mixnet_l",
        image_size=(640, 640),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type="",
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url="",  # no pretrained weights yet
    ),
    mobiledetv2_110d=dict(
        name="mobiledetv2_110d",
        backbone_name="mobilenetv2_110d",
        image_size=(384, 384),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=48,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        act_type="relu6",
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.05),
        url="",  # no pretrained weights yet
    ),
    mobiledetv2_120d=dict(
        name="mobiledetv2_120d",
        backbone_name="mobilenetv2_120d",
        image_size=(512, 512),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=56,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        act_type="relu6",
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.1),
        url="",  # no pretrained weights yet
    ),
    mobiledetv3_large=dict(
        name="mobiledetv3_large",
        backbone_name="mobilenetv3_large_100",
        image_size=(512, 512),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        act_type="hard_swish",
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.1),
        url="",  # no pretrained weights yet
    ),
    efficientdet_q0=dict(
        name="efficientdet_q0",
        backbone_name="efficientnet_b0",
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        fpn_name="qufpn_fa",  # quad-fpn + fast attn experiment
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.1),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_q0-bdf1bdb5.pth",
    ),
    efficientdet_q1=dict(
        name="efficientdet_q1",
        backbone_name="efficientnet_b1",
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        fpn_name="qufpn_fa",  # quad-fpn + fast attn experiment
        downsample_type="bilinear",
        upsample_type="bilinear",
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_q1b-d0612140.pth",
    ),
    efficientdet_q2=dict(
        name="efficientdet_q2",
        backbone_name="efficientnet_b2",
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type="",
        fpn_name="qufpn_fa",  # quad-fpn + fast attn experiment
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_q2-0f7564e5.pth",
    ),
    efficientdet_w0=dict(
        name="efficientdet_w0",  # 'wide'
        backbone_name="efficientnet_b0",
        image_size=(512, 512),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=80,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(
            drop_path_rate=0.1, feature_location="depthwise"
        ),  # features from after DW/SE in IR block
        url="",  # no pretrained weights yet
    ),
    efficientdet_es=dict(
        name="efficientdet_es",  # EdgeTPU-Small
        backbone_name="efficientnet_es",
        image_size=(512, 512),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=72,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        act_type="relu",
        redundant_bias=False,
        head_bn_level_first=True,
        separable_conv=False,
        backbone_args=dict(drop_path_rate=0.1),
        url="",
    ),
    efficientdet_em=dict(
        name="efficientdet_em",  # Edge-TPU Medium
        backbone_name="efficientnet_em",
        image_size=(640, 640),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=96,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type="",
        act_type="relu",
        redundant_bias=False,
        head_bn_level_first=True,
        separable_conv=False,
        backbone_args=dict(drop_path_rate=0.2),
        url="",  # no pretrained weights yet
    ),
    efficientdet_lite0=dict(
        name="efficientdet_lite0",
        backbone_name="efficientnet_lite0",
        image_size=(384, 384),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        act_type="relu6",
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.1),
        url="",
    ),
    # Models ported from Tensorflow with pretrained backbones ported from Tensorflow
    tf_efficientdet_d0=dict(
        name="tf_efficientdet_d0",
        backbone_name="tf_efficientnet_b0",
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_34-f153e0cf.pth",
    ),
    tf_efficientdet_d1=dict(
        name="tf_efficientdet_d1",
        backbone_name="tf_efficientnet_b1",
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d1_40-a30f94af.pth",
    ),
    tf_efficientdet_d2=dict(
        name="tf_efficientdet_d2",
        backbone_name="tf_efficientnet_b2",
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d2_43-8107aa99.pth",
    ),
    tf_efficientdet_d3=dict(
        name="tf_efficientdet_d3",
        backbone_name="tf_efficientnet_b3",
        image_size=(896, 896),
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3_47-0b525f35.pth",
    ),
    tf_efficientdet_d4=dict(
        name="tf_efficientdet_d4",
        backbone_name="tf_efficientnet_b4",
        image_size=(1024, 1024),
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4_49-f56376d9.pth",
    ),
    tf_efficientdet_d5=dict(
        name="tf_efficientdet_d5",
        backbone_name="tf_efficientnet_b5",
        image_size=(1280, 1280),
        fpn_channels=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5_51-c79f9be6.pth",
    ),
    tf_efficientdet_d6=dict(
        name="tf_efficientdet_d6",
        backbone_name="tf_efficientnet_b6",
        image_size=(1280, 1280),
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        fpn_name="bifpn_sum",  # Use unweighted sum for training stability.
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d6_52-4eda3773.pth",
    ),
    tf_efficientdet_d7=dict(
        name="tf_efficientdet_d7",
        backbone_name="tf_efficientnet_b6",
        image_size=(1536, 1536),
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        anchor_scale=5.0,
        fpn_name="bifpn_sum",  # Use unweighted sum for training stability.
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7_53-6d1d7a95.pth",
    ),
    tf_efficientdet_d7x=dict(
        name="tf_efficientdet_d7x",
        backbone_name="tf_efficientnet_b7",
        image_size=(1536, 1536),
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        anchor_scale=4.0,
        max_level=8,
        fpn_name="bifpn_sum",  # Use unweighted sum for training stability.
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7x-f390b87c.pth",
    ),
    #  Models ported from Tensorflow AdvProp+AA weights
    #  https://github.com/google/automl/blob/master/efficientdet/Det-AdvProp.md
    tf_efficientdet_d0_ap=dict(
        name="tf_efficientdet_d0_ap",
        backbone_name="tf_efficientnet_b0",
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_ap-d0cdbd0a.pth",
    ),
    tf_efficientdet_d1_ap=dict(
        name="tf_efficientdet_d1_ap",
        backbone_name="tf_efficientnet_b1",
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d1_ap-7721d075.pth",
    ),
    tf_efficientdet_d2_ap=dict(
        name="tf_efficientdet_d2_ap",
        backbone_name="tf_efficientnet_b2",
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d2_ap-a2995c19.pth",
    ),
    tf_efficientdet_d3_ap=dict(
        name="tf_efficientdet_d3_ap",
        backbone_name="tf_efficientnet_b3",
        image_size=(896, 896),
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3_ap-e4a2feab.pth",
    ),
    tf_efficientdet_d4_ap=dict(
        name="tf_efficientdet_d4_ap",
        backbone_name="tf_efficientnet_b4",
        image_size=(1024, 1024),
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4_ap-f601a5fc.pth",
    ),
    tf_efficientdet_d5_ap=dict(
        name="tf_efficientdet_d5_ap",
        backbone_name="tf_efficientnet_b5",
        image_size=(1280, 1280),
        fpn_channels=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5_ap-3673ae5d.pth",
    ),
    # The lite configs are in TF automl repository but no weights yet and listed as 'not final'
    tf_efficientdet_lite0=dict(
        name="tf_efficientdet_lite0",
        backbone_name="tf_efficientnet_lite0",
        image_size=(320, 320),
        anchor_scale=3.0,
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.0),
        **tf_efficientdet_lite_common,
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite0-dfacfc78.pth",
    ),
    tf_efficientdet_lite1=dict(
        name="tf_efficientdet_lite1",
        backbone_name="tf_efficientnet_lite1",
        image_size=(384, 384),
        anchor_scale=3.0,
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        **tf_efficientdet_lite_common,
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite1-6dc7ab30.pth",
    ),
    tf_efficientdet_lite2=dict(
        name="tf_efficientdet_lite2",
        backbone_name="tf_efficientnet_lite2",
        image_size=(448, 448),
        anchor_scale=3.0,
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        **tf_efficientdet_lite_common,
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite2-86c5d55b.pth",
    ),
    tf_efficientdet_lite3=dict(
        name="tf_efficientdet_lite3",
        backbone_name="tf_efficientnet_lite3",
        image_size=(512, 512),
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        **tf_efficientdet_lite_common,
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite3-506852cb.pth",
    ),
    tf_efficientdet_lite3x=dict(
        name="tf_efficientdet_lite3x",
        backbone_name="tf_efficientnet_lite3",
        image_size=(640, 640),
        anchor_scale=3.0,
        fpn_channels=200,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        **tf_efficientdet_lite_common,
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite3x-8404c57b.pth",
    ),
    tf_efficientdet_lite4=dict(
        name="tf_efficientdet_lite4",
        backbone_name="tf_efficientnet_lite4",
        image_size=(640, 640),
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        **tf_efficientdet_lite_common,
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite4-391ddabc.pth",
    ),
)


def get_efficientdet_config(model_name="tf_efficientdet_d1"):
    """Get the default config for EfficientDet based on model name."""
    h = default_detection_model_configs()
    h.update(efficientdet_model_param_dict[model_name])
    h.num_levels = h.max_level - h.min_level + 1
    h = deepcopy(h)  # may be unnecessary, ensure no references to param dict values
    # OmegaConf.set_struct(h, True)  # FIXME good idea?
    return h


def get_feat_sizes(image_size: Tuple[int, int], max_level: int):
    """Get feat widths and heights for all levels.
    Args:
      image_size: a tuple (H, W)
      max_level: maximum feature level.
    Returns:
      feat_sizes: a list of tuples (height, width) for each level.
    """
    feat_size = image_size
    feat_sizes = [feat_size]
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append(feat_size)
    return feat_sizes


def set_config_readonly(conf):
    OmegaConf.set_readonly(conf, True)


def set_config_writeable(conf):
    OmegaConf.set_readonly(conf, False)


def bifpn_config(min_level, max_level, weight_method=None):
    """BiFPN config.
    Adapted from https://github.com/google/automl/blob/56815c9986ffd4b508fe1d68508e268d129715c1/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or "fastattn"

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        p.nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": [level_last_id(i), level_last_id(i + 1)],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        p.nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": level_all_ids(i) + [level_last_id(i - 1)],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))
    return p


def panfpn_config(min_level, max_level, weight_method=None):
    """PAN FPN config.

    This defines FPN layout from Path Aggregation Networks as an alternate to
    BiFPN, it does not implement the full PAN spec.

    Paper: https://arxiv.org/abs/1803.01534
    """
    p = OmegaConf.create()
    weight_method = weight_method or "fastattn"

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level, min_level - 1, -1):
        # top-down path.
        offsets = (
            [level_last_id(i), level_last_id(i + 1)]
            if i != max_level
            else [level_last_id(i)]
        )
        p.nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": offsets,
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    for i in range(min_level, max_level + 1):
        # bottom-up path.
        offsets = (
            [level_last_id(i), level_last_id(i - 1)]
            if i != min_level
            else [level_last_id(i)]
        )
        p.nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": offsets,
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    return p


def qufpn_config(min_level, max_level, weight_method=None):
    """A dynamic quad fpn config that can adapt to different min/max levels.

    It extends the idea of BiFPN, and has four paths:
        (up_down -> bottom_up) + (bottom_up -> up_down).

    Paper: https://ieeexplore.ieee.org/document/9225379
    Ref code: From contribution to TF EfficientDet
    https://github.com/google/automl/blob/eb74c6739382e9444817d2ad97c4582dbe9a9020/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or "fastattn"
    quad_method = "fastattn"
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    level_first_id = lambda level: node_ids[level][0]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path 1.
        p.nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": [level_last_id(i), level_last_id(i + 1)],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    for i in range(min_level + 1, max_level):
        # bottom-up path 2.
        p.nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": level_all_ids(i) + [level_last_id(i - 1)],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    i = max_level
    p.nodes.append(
        {
            "feat_level": i,
            "inputs_offsets": [level_first_id(i)] + [level_last_id(i - 1)],
            "weight_method": weight_method,
        }
    )
    node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(min_level + 1, max_level + 1, 1):
        # bottom-up path 3.
        p.nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": [
                    level_first_id(i),
                    (
                        level_last_id(i - 1)
                        if i != min_level + 1
                        else level_first_id(i - 1)
                    ),
                ],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(max_level - 1, min_level, -1):
        # top-down path 4.
        p.nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": [node_ids[i][0]]
                + [node_ids[i][-1]]
                + [level_last_id(i + 1)],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))
    i = min_level
    p.nodes.append(
        {
            "feat_level": i,
            "inputs_offsets": [node_ids[i][0]] + [level_last_id(i + 1)],
            "weight_method": weight_method,
        }
    )
    node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    # NOTE: the order of the quad path is reversed from the original, my code expects the output of
    # each FPN repeat to be same as input from backbone, in order of increasing reductions
    for i in range(min_level, max_level + 1):
        # quad-add path.
        p.nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": [node_ids[i][2], node_ids[i][4]],
                "weight_method": quad_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    return p


def get_fpn_config(fpn_name, min_level=3, max_level=7):
    if not fpn_name:
        fpn_name = "bifpn_fa"
    name_to_config = {
        "bifpn_sum": bifpn_config(
            min_level=min_level, max_level=max_level, weight_method="sum"
        ),
        "bifpn_attn": bifpn_config(
            min_level=min_level, max_level=max_level, weight_method="attn"
        ),
        "bifpn_fa": bifpn_config(
            min_level=min_level, max_level=max_level, weight_method="fastattn"
        ),
        "pan_sum": panfpn_config(
            min_level=min_level, max_level=max_level, weight_method="sum"
        ),
        "pan_fa": panfpn_config(
            min_level=min_level, max_level=max_level, weight_method="fastattn"
        ),
        "qufpn_sum": qufpn_config(
            min_level=min_level, max_level=max_level, weight_method="sum"
        ),
        "qufpn_fa": qufpn_config(
            min_level=min_level, max_level=max_level, weight_method="fastattn"
        ),
    }
    return name_to_config[fpn_name]


class SequentialList(nn.Sequential):
    """This module exists to work around torchscript typing issues list -> list"""

    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding="",
        bias=False,
        norm_layer=nn.BatchNorm2d,
        act_layer=_ACT_LAYER,
    ):
        super(ConvBnAct2d, self).__init__()
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """Separable Conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding="",
        bias=False,
        channel_multiplier=1.0,
        pw_kernel_size=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=_ACT_LAYER,
    ):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(
            in_channels,
            int(in_channels * channel_multiplier),
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            depthwise=True,
        )
        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier),
            out_channels,
            pw_kernel_size,
            padding=padding,
            bias=bias,
        )
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Interpolate2d(nn.Module):
    r"""Resamples a 2d Image

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """

    __constants__ = ["size", "scale_factor", "mode", "align_corners", "name"]
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ) -> None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == "nearest" else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=False,
        )


class ResampleFeatureMap(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        input_size,
        output_size,
        pad_type="",
        downsample=None,
        upsample=None,
        norm_layer=nn.BatchNorm2d,
        apply_bn=False,
        redundant_bias=False,
    ):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or "max"
        upsample = upsample or "nearest"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size

        if in_channels != out_channels:
            self.add_module(
                "conv",
                ConvBnAct2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=pad_type,
                    norm_layer=norm_layer if apply_bn else None,
                    bias=not apply_bn or redundant_bias,
                    act_layer=None,
                ),
            )

        if input_size[0] > output_size[0] and input_size[1] > output_size[1]:
            if downsample in ("max", "avg"):
                stride_size_h = int((input_size[0] - 1) // output_size[0] + 1)
                stride_size_w = int((input_size[1] - 1) // output_size[1] + 1)
                if stride_size_h == stride_size_w:
                    kernel_size = stride_size_h + 1
                    stride = stride_size_h
                else:
                    # FIXME need to support tuple kernel / stride input to padding fns
                    kernel_size = (stride_size_h + 1, stride_size_w + 1)
                    stride = (stride_size_h, stride_size_w)
                down_inst = create_pool2d(
                    downsample, kernel_size=kernel_size, stride=stride, padding=pad_type
                )
            else:
                if (
                    _USE_SCALE
                ):  # FIXME not sure if scale vs size is better, leaving both in to test for now
                    scale = (
                        output_size[0] / input_size[0],
                        output_size[1] / input_size[1],
                    )
                    down_inst = Interpolate2d(scale_factor=scale, mode=downsample)
                else:
                    down_inst = Interpolate2d(size=output_size, mode=downsample)
            self.add_module("downsample", down_inst)
        else:
            if input_size[0] < output_size[0] or input_size[1] < output_size[1]:
                if _USE_SCALE:
                    scale = (
                        output_size[0] / input_size[0],
                        output_size[1] / input_size[1],
                    )
                    self.add_module(
                        "upsample", Interpolate2d(scale_factor=scale, mode=upsample)
                    )
                else:
                    self.add_module(
                        "upsample", Interpolate2d(size=output_size, mode=upsample)
                    )


class FpnCombine(nn.Module):
    def __init__(
        self,
        feature_info,
        fpn_channels,
        inputs_offsets,
        output_size,
        pad_type="",
        downsample=None,
        upsample=None,
        norm_layer=nn.BatchNorm2d,
        apply_resample_bn=False,
        redundant_bias=False,
        weight_method="attn",
    ):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            self.resample[str(offset)] = ResampleFeatureMap(
                feature_info[offset]["num_chs"],
                fpn_channels,
                input_size=feature_info[offset]["size"],
                output_size=output_size,
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_bn=apply_resample_bn,
                redundant_bias=redundant_bias,
            )

        if weight_method == "attn" or weight_method == "fastattn":
            self.edge_weights = nn.Parameter(
                torch.ones(len(inputs_offsets)), requires_grad=True
            )  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == "attn":
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == "fastattn":
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [
                    (nodes[i] * edge_weights[i]) / (weights_sum + 0.0001)
                    for i in range(len(nodes))
                ],
                dim=-1,
            )
        elif self.weight_method == "sum":
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError("unknown weight_method {}".format(self.weight_method))
        out = torch.sum(out, dim=-1)
        return out


class Fnode(nn.Module):
    """A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """

    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class BiFpnLayer(nn.Module):
    def __init__(
        self,
        feature_info,
        feat_sizes,
        fpn_config,
        fpn_channels,
        num_levels=5,
        pad_type="",
        downsample=None,
        upsample=None,
        norm_layer=nn.BatchNorm2d,
        act_layer=_ACT_LAYER,
        apply_resample_bn=False,
        pre_act=True,
        separable_conv=True,
        redundant_bias=False,
    ):
        super(BiFpnLayer, self).__init__()
        self.num_levels = num_levels
        # fill feature info for all FPN nodes (chs and feat size) before creating FPN nodes
        fpn_feature_info = feature_info + [
            dict(num_chs=fpn_channels, size=feat_sizes[fc["feat_level"]])
            for fc in fpn_config.nodes
        ]

        self.fnode = nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug("fnode {} : {}".format(i, fnode_cfg))
            combine = FpnCombine(
                fpn_feature_info,
                fpn_channels,
                tuple(fnode_cfg["inputs_offsets"]),
                output_size=feat_sizes[fnode_cfg["feat_level"]],
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_resample_bn=apply_resample_bn,
                redundant_bias=redundant_bias,
                weight_method=fnode_cfg["weight_method"],
            )

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels,
                out_channels=fpn_channels,
                kernel_size=3,
                padding=pad_type,
                bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            if pre_act:
                conv_kwargs["bias"] = redundant_bias
                conv_kwargs["act_layer"] = None
                after_combine.add_module("act", act_layer(inplace=True))
            after_combine.add_module(
                "conv",
                (
                    SeparableConv2d(**conv_kwargs)
                    if separable_conv
                    else ConvBnAct2d(**conv_kwargs)
                ),
            )

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))

        self.feature_info = fpn_feature_info[-num_levels::]

    def forward(self, x: List[torch.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels : :]


class BiFpn(nn.Module):
    def __init__(self, config, feature_info):
        super(BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(
            config.fpn_name, min_level=config.min_level, max_level=config.max_level
        )

        feat_sizes = get_feat_sizes(config.image_size, max_level=config.max_level)
        prev_feat_size = feat_sizes[config.min_level]
        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            feat_size = feat_sizes[level + config.min_level]
            if level < len(feature_info):
                in_chs = feature_info[level]["num_chs"]
                feature_info[level]["size"] = feat_size
            else:
                # Adds a coarser level by downsampling the last feature map
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    input_size=prev_feat_size,
                    output_size=feat_size,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    apply_bn=config.apply_resample_bn,
                    redundant_bias=config.redundant_bias,
                )
                in_chs = config.fpn_channels
                feature_info.append(dict(num_chs=in_chs, size=feat_size))
            prev_feat_size = feat_size

        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug("building cell {}".format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                feat_sizes=feat_sizes,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                pre_act=not config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x


class HeadNet(nn.Module):
    def __init__(self, config, num_outputs):
        super(HeadNet, self).__init__()
        self.num_levels = config.num_levels
        self.bn_level_first = getattr(config, "head_bn_level_first", False)
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_type = (
            config.head_act_type
            if getattr(config, "head_act_type", None)
            else config.act_type
        )
        act_layer = get_act_layer(act_type) or _ACT_LAYER

        # Build convolution repeats
        conv_fn = SeparableConv2d if config.separable_conv else ConvBnAct2d
        conv_kwargs = dict(
            in_channels=config.fpn_channels,
            out_channels=config.fpn_channels,
            kernel_size=3,
            padding=config.pad_type,
            bias=config.redundant_bias,
            act_layer=None,
            norm_layer=None,
        )
        self.conv_rep = nn.ModuleList(
            [conv_fn(**conv_kwargs) for _ in range(config.box_class_repeats)]
        )

        # Build batchnorm repeats. There is a unique batchnorm per feature level for each repeat.
        # This can be organized with repeats first or feature levels first in module lists, the original models
        # and weights were setup with repeats first, levels first is required for efficient torchscript usage.
        self.bn_rep = nn.ModuleList()
        if self.bn_level_first:
            for _ in range(self.num_levels):
                self.bn_rep.append(
                    nn.ModuleList(
                        [
                            norm_layer(config.fpn_channels)
                            for _ in range(config.box_class_repeats)
                        ]
                    )
                )
        else:
            for _ in range(config.box_class_repeats):
                self.bn_rep.append(
                    nn.ModuleList(
                        [
                            nn.Sequential(
                                OrderedDict([("bn", norm_layer(config.fpn_channels))])
                            )
                            for _ in range(self.num_levels)
                        ]
                    )
                )

        self.act = act_layer(inplace=True)

        # Prediction (output) layer. Has bias with special init reqs, see init fn.
        num_anchors = len(config.aspect_ratios) * config.num_scales
        predict_kwargs = dict(
            in_channels=config.fpn_channels,
            out_channels=num_outputs * num_anchors,
            kernel_size=3,
            padding=config.pad_type,
            bias=True,
            norm_layer=None,
            act_layer=None,
        )
        self.predict = conv_fn(**predict_kwargs)

    @torch.jit.ignore()
    def toggle_bn_level_first(self):
        """Toggle the batchnorm layers between feature level first vs repeat first access pattern
        Limitations in torchscript require feature levels to be iterated over first.

        This function can be used to allow loading weights in the original order, and then toggle before
        jit scripting the model.
        """
        with torch.no_grad():
            new_bn_rep = nn.ModuleList()
            for i in range(len(self.bn_rep[0])):
                bn_first = nn.ModuleList()
                for r in self.bn_rep.children():
                    m = r[i]
                    # NOTE original rep first model def has extra Sequential container with 'bn', this was
                    # flattened in the level first definition.
                    bn_first.append(
                        m[0]
                        if isinstance(m, nn.Sequential)
                        else nn.Sequential(OrderedDict([("bn", m)]))
                    )
                new_bn_rep.append(bn_first)
            self.bn_level_first = not self.bn_level_first
            self.bn_rep = new_bn_rep

    @torch.jit.ignore()
    def _forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for level in range(self.num_levels):
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, self.bn_rep):
                x_level = conv(x_level)
                x_level = bn[level](x_level)  # this is not allowed in torchscript
                x_level = self.act(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def _forward_level_first(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for level, bn_rep in enumerate(
            self.bn_rep
        ):  # iterating over first bn dim first makes TS happy
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, bn_rep):
                x_level = conv(x_level)
                x_level = bn(x_level)
                x_level = self.act(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.bn_level_first:
            return self._forward_level_first(x)
        else:
            return self._forward(x)


def _init_weight(
    m,
    n="",
):
    """Weight initialization as per Tensorflow official implementations."""

    def _fan_in_out(w, groups=1):
        dimensions = w.dim()
        if dimensions < 2:
            raise ValueError(
                "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
            )
        num_input_fmaps = w.size(1)
        num_output_fmaps = w.size(0)
        receptive_field_size = 1
        if w.dim() > 2:
            receptive_field_size = w[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        fan_out //= groups
        return fan_in, fan_out

    def _glorot_uniform(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1.0, (fan_in + fan_out) / 2.0)  # fan avg
        limit = math.sqrt(3.0 * gain)
        w.data.uniform_(-limit, limit)

    def _variance_scaling(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1.0, fan_in)  # fan in
        # gain /= max(1., (fan_in + fan_out) / 2.)  # fan

        # should it be normal or trunc normal? using normal for now since no good trunc in PT
        # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        # std = math.sqrt(gain) / .87962566103423978
        # w.data.trunc_normal(std=std)
        std = math.sqrt(gain)
        w.data.normal_(std=std)

    if isinstance(m, SeparableConv2d):
        if "box_net" in n or "class_net" in n:
            _variance_scaling(m.conv_dw.weight, groups=m.conv_dw.groups)
            _variance_scaling(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                if "class_net.predict" in n:
                    m.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pw.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_dw.weight, groups=m.conv_dw.groups)
            _glorot_uniform(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                m.conv_pw.bias.data.zero_()
    elif isinstance(m, ConvBnAct2d):
        if "box_net" in n or "class_net" in n:
            m.conv.weight.data.normal_(std=0.01)
            if m.conv.bias is not None:
                if "class_net.predict" in n:
                    m.conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv.bias.data.zero_()
        else:
            _glorot_uniform(m.conv.weight)
            if m.conv.bias is not None:
                m.conv.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        # looks like all bn init the same?
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def _init_weight_alt(
    m,
    n="",
):
    """Weight initialization alternative, based on EfficientNet bacbkone init w/ class bias addition
    NOTE: this will likely be removed after some experimentation
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            if "class_net.predict" in n:
                m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
            else:
                m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def get_feature_info(backbone):
    if isinstance(backbone.feature_info, Callable):
        # old accessor for timm versions <= 0.1.30, efficientnet and mobilenetv3 and related nets only
        feature_info = [
            dict(num_chs=f["num_chs"], reduction=f["reduction"])
            for i, f in enumerate(backbone.feature_info())
        ]
    else:
        # new feature info accessor, timm >= 0.2, all models supported
        feature_info = backbone.feature_info.get_dicts(keys=["num_chs", "reduction"])
    return feature_info


class EfficientDet(nn.Module):
    def __init__(
        self,
        config,
        pretrained_backbone=True,
        alternate_init=False,
        compression_method=RandomCompression,
        compression_parameters=None,
        is_compressed=False,
    ):
        super(EfficientDet, self).__init__()
        self.config = config
        set_config_readonly(self.config)
        self.backbone = create_model(
            config.backbone_name,
            features_only=True,
            out_indices=self.config.backbone_indices or (2, 3, 4),
            pretrained=pretrained_backbone,
            **config.backbone_args,
        )
        feature_info = get_feature_info(self.backbone)
        self.fpn = BiFpn(self.config, feature_info)
        self.class_net = HeadNet(self.config, num_outputs=self.config.num_classes)
        self.box_net = HeadNet(self.config, num_outputs=4)

        for n, m in self.named_modules():
            if "backbone" not in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

        # Compression
        self.compression_method = compression_method
        self.compression_parameters = compression_parameters
        self.compression = compression_method(self, compression_parameters)
        self.is_compressed = is_compressed

        WCC_wavelet(self.backbone, levels=3, compress_rate=0.5, bit_w=4, bit_a=8)

        a = 2

    @torch.jit.ignore()
    def reset_head(
        self,
        num_classes=None,
        aspect_ratios=None,
        num_scales=None,
        alternate_init=False,
    ):
        reset_class_head = False
        reset_box_head = False
        set_config_writeable(self.config)
        if num_classes is not None:
            reset_class_head = True
            self.config.num_classes = num_classes
        if aspect_ratios is not None:
            reset_box_head = True
            self.config.aspect_ratios = aspect_ratios
        if num_scales is not None:
            reset_box_head = True
            self.config.num_scales = num_scales
        set_config_readonly(self.config)

        if reset_class_head:
            self.class_net = HeadNet(self.config, num_outputs=self.config.num_classes)
            for n, m in self.class_net.named_modules(prefix="class_net"):
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

        if reset_box_head:
            self.box_net = HeadNet(self.config, num_outputs=4)
            for n, m in self.box_net.named_modules(prefix="box_net"):
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    @torch.jit.ignore()
    def toggle_head_bn_level_first(self):
        """Toggle the head batchnorm layers between being access with feature_level first vs repeat"""
        self.class_net.toggle_bn_level_first()
        self.box_net.toggle_bn_level_first()

    def forward(self, x):
        context = self.is_compressed
        with ConditionalContext(context, self.compression):
            x = self.backbone(x)
        x = self.fpn(x)
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box
