import torch
import torch.nn as nn
from torchsummary import summary
import argparse
import engine.utils.memory_utils as memory_utils
from engine.compression import (
    WaveletTransformCompression,
    RandomCompression,
    ThresholdingCompression,
    DCTCompression,
)
from rich import print

from models.clasification import *
from models.segmentation import *
from models.clasification.deeplabv3 import deeplabv3plus_mobilenet
from models.clasification.edsr import EDSR
from models.clasification.efficientnet import EfficientNet

from engine.config import config

from WaveletCompressedConvolution.WCC.transform_model import (
    wavelet_module as WCC_wavelet,
)

from engine.memory_test import MemoryTest, prune_model


def main_efficientnet(
    memory_type,
    block_args=config.block_args_EfficientNetB0,
):
    """
    We are not using Lenet because using WT doesn't compress at all
    """

    if block_args == config.block_args_EfficientNetB0:
        batch_size = 4
    elif block_args == config.block_args_EfficientNetB4:
        batch_size = 2
    elif block_args == config.block_args_EfficientNetB8:
        batch_size = 1

    width = height = 640
    n_channels = 3
    n_class = 1000

    use_half = False
    is_ckpt = False
    segments = 1

    M = MemoryTest(
        batch_size=batch_size,
        n_channels=n_channels,
        width=width,
        height=height,
        depth=None,
        n_class=n_class,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    if memory_type == "baseline":
        net = EfficientNet(block_args=block_args)
    elif memory_type == "others":
        net = EfficientNet(
            block_args=block_args,
            compression_method=RandomCompression,
            is_compressed=True,
        )
    elif memory_type == "ckpt":
        net = EfficientNet(block_args=block_args)
        is_ckpt = True
    elif memory_type == "quan":
        net = EfficientNet(block_args=block_args)
        use_half = True
    elif memory_type == "wcc":
        net = EfficientNet(block_args=block_args)
        for compress_rate in [
            1,
            # 0.5,
            # 0.25,
            # 0.125,
            # 0.0625,
        ]:
            WCC_wavelet(net, levels=3, compress_rate=compress_rate, bit_w=4, bit_a=8)
    else:
        for compression_ratio in [
            # 0.1,
            # 0.3,
            # 0.6,
            # 0.9,
            # 0.99,
            0.999,
        ]:
            net = EfficientNet(
                block_args=block_args,
                compression_method=DCTCompression,
                compression_parameters={"compression_ratio": compression_ratio},
                is_compressed=True,
            )

    M.perform_memory_test_classification(
        net=net.cuda(),
        print_text=memory_type,
        use_half=use_half,
        is_ckpt=is_ckpt,
        segments=segments,
    )

    print()
    print()
    print()
    print()


def main_efficientdet(
    memory_type,
):
    """
    We are not using Lenet because using WT doesn't compress at all
    """

    width = height = 640
    n_channels = 3
    n_class = 80
    batch_size = 4

    # width = height = 640
    # batch_size = 1
    # n_channels = 3
    # n_class = 1000

    M = MemoryTest(
        batch_size=batch_size,
        n_channels=n_channels,
        width=width,
        height=height,
        depth=None,
        n_class=n_class,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    if memory_type == "baseline":
        net = EfficientDet(
            config=get_efficientdet_config(model_name="tf_efficientdet_d1"),
            pretrained_backbone=False,
        )
    elif memory_type == "others":
        net = EfficientDet(
            config=get_efficientdet_config(model_name="tf_efficientdet_d1"),
            pretrained_backbone=False,
            compression_method=RandomCompression,
            is_compressed=True,
        )
    elif memory_type == "wcc":
        net = EfficientDet(
            config=get_efficientdet_config(model_name="tf_efficientdet_d1"),
            pretrained_backbone=False,
        )
        for compress_rate in [
            # 1,
            0.5,
            # 0.25,
            # 0.125,
            # 0.0625,
        ]:
            WCC_wavelet(net, levels=3, compress_rate=compress_rate, bit_w=4, bit_a=8)
    else:
        for compression_ratio in [
            # 0.1,
            # 0.3,
            # 0.6,
            # 0.9,
            # 0.99,
            0.999,
        ]:
            net = EfficientDet(
                config=get_efficientdet_config(model_name="tf_efficientdet_d1"),
                pretrained_backbone=False,
                compression_method=DCTCompression,
                compression_parameters={"compression_ratio": compression_ratio},
                is_compressed=True,
            )

    M.perform_memory_test_classification(
        net=net.cuda(),
        print_text=memory_type,
    )

    print()
    print()
    print()
    print()


def main_deeplab(memory_type):
    """
    We are not using Lenet because using WT doesn't compress at all
    """

    width = 128
    height = 128
    n_channels = 3
    n_class = 100
    batch_size = 32

    # width = height = 640
    # batch_size = 1
    # n_channels = 3
    # n_class = 1000

    M = MemoryTest(
        batch_size=batch_size,
        n_channels=n_channels,
        width=width,
        height=height,
        depth=None,
        n_class=n_class,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    use_half = False
    is_ckpt = False
    segments = 2

    if memory_type == "baseline":
        net = deeplabv3plus_mobilenet(pretrained_backbone=False)
    elif memory_type == "others":
        net = deeplabv3plus_mobilenet(
            pretrained_backbone=False,
            compression_method=RandomCompression,
            is_compressed=True,
        )
    elif memory_type == "quan":
        net = deeplabv3plus_mobilenet(pretrained_backbone=False)
        use_half = True
    elif memory_type == "ckpt":
        net = deeplabv3plus_mobilenet(pretrained_backbone=False)
        is_ckpt = True
    elif memory_type == "prune":
        net = deeplabv3plus_mobilenet(pretrained_backbone=False)
        prune_model(net, amount=0.9)
    elif memory_type == "wcc":
        net = deeplabv3plus_mobilenet(pretrained_backbone=False)
        for compress_rate in [
            # 1,
            # 0.5,
            # 0.25,
            # 0.125,
            0.0625,
        ]:
            WCC_wavelet(net, levels=3, compress_rate=compress_rate, bit_w=4, bit_a=8)
    else:
        for compression_ratio in [
            # 0.1,
            # 0.3,
            # 0.6,
            # 0.9,
            # 0.99,
            0.999,
        ]:
            net = deeplabv3plus_mobilenet(
                pretrained_backbone=False,
                compression_method=DCTCompression,
                compression_parameters={"compression_ratio": compression_ratio},
                is_compressed=True,
            )

    M.perform_memory_test_classification(
        net=net.cuda(),
        print_text=memory_type,
        use_half=use_half,
        is_ckpt=is_ckpt,
        segments=segments,
    )

    print()
    print()
    print()
    print()


def main_edsr(
    memory_type,
):
    """
    We are not using Lenet because using WT doesn't compress at all
    """

    parser = argparse.ArgumentParser(description="EDSR and MDSR")
    parser.add_argument("--scale", type=str, default="4", help="super resolution scale")
    parser.add_argument(
        "--rgb_range", type=int, default=255, help="maximum value of RGB"
    )
    parser.add_argument(
        "--n_colors", type=int, default=3, help="number of color channels to use"
    )
    args = parser.parse_args()

    args.scale = list(map(lambda x: int(x), args.scale.split("+")))
    args.model = "EDSR"
    args.n_resblocks = 32
    args.n_feats = 256
    args.res_scale = 0.1

    width = 48
    height = 48
    n_channels = 3
    n_class = 100
    batch_size = 4

    use_half = False
    is_ckpt = False
    segments = 5

    M = MemoryTest(
        batch_size=batch_size,
        n_channels=n_channels,
        width=width,
        height=height,
        depth=None,
        n_class=n_class,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    if memory_type == "baseline":
        net = EDSR(args)
    elif memory_type == "others":
        net = EDSR(args, compression_method=RandomCompression, is_compressed=True)
    elif memory_type == "quan":
        net = EDSR(args)
        use_half = True
    elif memory_type == "ckpt":
        net = EDSR(args)
        is_ckpt = True
    elif memory_type == "wcc":
        net = EDSR(args)
        for compress_rate in [
            # 1,
            # 0.5,
            # 0.25,
            # 0.125,
            0.0625,
        ]:
            WCC_wavelet(net, levels=3, compress_rate=compress_rate, bit_w=4, bit_a=8)
    else:
        for compression_ratio in [
            # 0.1,
            # 0.3,
            # 0.6,
            # 0.9,
            # 0.99,
            0.999,
        ]:
            net = EDSR(
                args,
                compression_method=DCTCompression,
                compression_parameters={"compression_ratio": compression_ratio},
                is_compressed=True,
            )

    M.perform_memory_test_classification(
        net=net.cuda(),
        print_text=memory_type,
        use_half=use_half,
        is_ckpt=is_ckpt,
        segments=segments,
    )

    print()
    print()
    print()
    print()


def main_resnet18(memory_type):
    """
    We are not using Lenet because using WT doesn't compress at all
    """

    M = MemoryTest(
        batch_size=128,
        n_channels=3,
        width=32,
        height=32,
        depth=None,
        n_class=100,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    ## Mem. Baseline
    if memory_type == "baseline":
        M.perform_memory_test_classification(
            net=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100).cuda(),
            print_text="without compression",
        )

    ## Mem. Others
    elif memory_type == "others":
        M.perform_memory_test_classification(
            net=ResNetCompressed(
                num_classes=100,
                is_compressed=[True, True, True, True, True],
                compression_method=RandomCompression,
                compressed_layers_layer0=["conv1", "bn1", "relu1"],
                compressed_layers_layer1_4=[
                    "conv1",
                    "bn1",
                    "relu1",
                    "conv2",
                    "bn2",
                    "relu2",
                ],
            ).cuda(),
            print_text="without compression",
        )

    ## Mem.
    else:
        for is_compressed in [
            # [True, False, False, False, False],
            [True, True, False, False, False],
            # [True, True, True, False, False],
            # [True, True, True, True, False],
            # [True, True, True, True, True],
        ]:
            for compressed_layers_layer0, compressed_layers_layer1_4 in zip(
                [
                    []
                    # ["bn1"],
                    # ["conv1", "bn1"],
                    # ["conv1", "bn1", "relu1"],
                ],
                [
                    # ["conv1"],
                    # ["conv1", "bn1"],
                    # ["conv1", "bn1", "relu1"],
                    # ["conv1", "bn1", "relu1", "conv2"],
                    # ["conv1", "bn1", "relu1", "conv2", "bn2"],
                    # ["bn1", "bn2"],
                    # ["conv1", "conv2"],
                    # ["conv1", "bn1", "conv2", "bn2"],
                    # ["bn1", "relu1", "bn2", "relu2"],
                    # ["conv1", "bn1", "relu1", "conv2", "bn2"],
                    # ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
                    ["conv1", "bn1", "relu1", "conv2", "relu2"],
                ],
            ):
                print()
                print()
                print()
                print()
                print(is_compressed)
                print(compressed_layers_layer0)
                print(compressed_layers_layer1_4)

                # # For WT + energy (will used in the follow-up paper)
                # M.perform_memory_test_classification(
                #     net=ResNetCompressed(
                #         num_classes=100,
                #         compression_method=WaveletTransformCompression,
                #         is_compressed=is_compressed,
                #         compression_parameters={
                #             "wave": "db3",
                #             "compression_ratio": 0.9,
                #             "n_levels": 3,
                #         },
                #         compressed_layers_layer0=compressed_layers_layer0,
                #         compressed_layers_layer1_4=compressed_layers_layer1_4,
                #     ).cuda(),
                #     print_text="with compression",
                # )

                M.perform_memory_test_classification(
                    net=ResNetCompressed(
                        num_classes=100,
                        compression_method=DCTCompression,
                        is_compressed=is_compressed,
                        compression_parameters={
                            "compression_ratio": 0.99,
                        },
                        compressed_layers_layer0=compressed_layers_layer0,
                        compressed_layers_layer1_4=compressed_layers_layer1_4,
                    ).cuda(),
                    print_text="with compression",
                )


def main():
    # main_resnet18("others")
    # main_resnet18("baseline")
    main_resnet18("")

    # main_unet("others")
    # main_unet("baseline")
    # main_unet("")

    # main_resnet152_basic("baseline")
    # main_resnet152_basic("others")
    # main_resnet152_basic("")

    # main_resnet152_bottleneck("baseline")
    # main_resnet152_bottleneck("others")
    # main_resnet152_bottleneck("")

    # main_wideresnet101_bottleneck("baseline")
    # main_wideresnet101_bottleneck("others")
    # main_wideresnet101_bottleneck("")

    # main_vit("baseline")
    # main_vit("others")
    # main_vit("")

    # main_vit3d("baseline")
    # main_vit3d("others")
    # main_vit3d("")

    # main_efficientdet("others")
    # main_efficientdet("baseline")
    # main_efficientdet("wcc")
    # main_efficientdet("ours")

    for block_args in [
        # config.block_args_EfficientNetB0,
        config.block_args_EfficientNetB4,
        # config.block_args_EfficientNetB8,
    ]:
        # main_efficientnet("others", block_args=block_args)
        # main_efficientnet("baseline", block_args=block_args)
        # main_efficientnet("wcc", block_args=block_args)
        # main_efficientnet("ours", block_args=block_args)
        main_efficientnet("quan", block_args=block_args)
        main_efficientnet("ckpt", block_args=block_args)

    main_deeplab("others")
    # main_deeplab("baseline")
    # main_deeplab("wcc")
    # main_deeplab("ours")
    # main_deeplab("quan")
    # main_deeplab("prune")
    # main_deeplab("ckpt")

    # main_edsr("others")
    # main_edsr("baseline")
    # main_edsr("wcc")
    # main_edsr("ours")
    main_edsr("quan")
    main_edsr("ckpt")

    print()
    print()
    print()
    print()
    print()
    print()


if __name__ == "__main__":
    # test()
    # main_mnist()
    # main_lenet()
    # main()

    main_resnet18("")

    print()
