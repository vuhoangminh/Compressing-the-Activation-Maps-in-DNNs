import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.checkpoint import checkpoint_sequential
import engine.utils.memory_utils as memory_utils
from engine.compression import (
    WaveletTransformCompression,
    RandomCompression,
    ThresholdingCompression,
    DCTCompression,
)

from models.clasification import *
from models.segmentation import *

from models.vit.vit import ViT, ViTCompressed
from models.vit.vit_3d import ViT as ViT3d
from models.vit.vit_3d import ViTCompressed as ViTCompressed3d
from monai.losses import DiceLoss


import torch.nn.utils.prune as prune


def prune_model(model, amount=0.2):
    # Iterate over all modules in the model
    for name, module in model.named_modules():
        # Check if the module has a weight attribute
        if hasattr(module, "weight"):
            # Prune the weights
            prune.l1_unstructured(module, name="weight", amount=amount)
            # Make the pruning permanent
            prune.remove(module, "weight")


def convert_to_sequential(model):
    def model_to_list(model, modules):
        if len(list(model.children())) == 0:
            modules.append(model)
            return
        for module in model.children():
            model_to_list(module, modules)

    modules = []
    model_to_list(model, modules)
    return nn.Sequential(*modules)


class MemoryTest:
    def __init__(
        self,
        batch_size,
        n_channels,
        width,
        height,
        frames=16,  # ViT
        n_class=10,
        depth=None,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    ):
        super(MemoryTest, self).__init__()
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.width = width
        self.height = height
        self.depth = depth
        self.n_class = n_class
        self.frames = frames
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.verbose = verbose

    def perform_memory_test_classification(
        self,
        net,
        use_half=False,
        is_ckpt=False,
        segments=2,  # ckpt
        print_text="without compression",
    ):
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(
            net.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4
        )

        # if self.verbose:
        #     if self.depth is None:
        #         summary(net, (self.n_channels, self.width, self.height))

        print("=" * 60)
        print(print_text)
        print("=" * 60)

        time_list = list()
        for epoch in range(self.n_epoch):
            start_time = time.time()
            for batch in range(self.n_batch):
                if self.verbose:
                    print(
                        "Epoch {} --- Batch {} --- before forward".format(epoch, batch)
                    )

                torch.cuda.empty_cache()
                # memory_utils.print_memory(verbose=1)

                x = torch.randn(
                    self.batch_size, self.n_channels, self.width, self.height
                ).cuda()

                target = torch.zeros(self.batch_size).type(torch.LongTensor).cuda()

                if use_half:
                    net = net.half()
                    x = x.half()

                if is_ckpt:
                    net = convert_to_sequential(net)
                    modules = [module for k, module in net._modules.items()]
                    y = checkpoint_sequential(modules, segments, x)
                else:
                    y = net(x)

                if self.verbose:
                    print(
                        "Epoch {} --- Batch {} --- after forward".format(epoch, batch)
                    )

                torch.cuda.empty_cache()
                _, _, memory_allocated, _ = memory_utils.print_memory(verbose=1)

                # compute output
                loss = criterion(y, target)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.verbose:
                    print(
                        "Epoch {} --- Batch {} --- after backward".format(epoch, batch)
                    )

                torch.cuda.empty_cache()
                # memory_utils.print_memory(verbose=1)

            end_time = time.time() - start_time
            time_list.append(end_time)

        print(time_list)

        print()
        print()
        print(f"Mem. allocated: {memory_allocated:.0f}")

    def perform_memory_test_segmentation(
        self,
        net,
        print_text="without compression",
    ):
        criterion = DiceLoss().cuda()
        optimizer = torch.optim.SGD(
            net.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4
        )

        # if self.verbose:
        #     if self.depth is None:
        #         summary(net, (self.n_channels, self.width, self.height))
        #     else:
        #         summary(net, (self.n_channels, self.width, self.height, self.depth))

        print("=" * 60)
        print(print_text)
        print("=" * 60)

        time_list = list()
        for epoch in range(self.n_epoch):
            start_time = time.time()
            for batch in range(self.n_batch):
                if self.verbose:
                    print(
                        "Epoch {} --- Batch {} --- before forward".format(epoch, batch)
                    )

                torch.cuda.empty_cache()
                # memory_utils.print_memory(verbose=1)

                x = torch.randn(
                    self.batch_size,
                    self.n_channels,
                    self.width,
                    self.height,
                    self.depth,
                ).cuda()

                target = (
                    torch.zeros(
                        self.batch_size,
                        self.n_class,
                        self.width,
                        self.height,
                        self.depth,
                    )
                    .type(torch.LongTensor)
                    .cuda()
                )

                y = net(x)

                if self.verbose:
                    print(
                        "Epoch {} --- Batch {} --- after forward".format(epoch, batch)
                    )

                torch.cuda.empty_cache()
                memory_utils.print_memory(verbose=1)

                # compute output
                loss = criterion(y, target)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.verbose:
                    print(
                        "Epoch {} --- Batch {} --- after backward".format(epoch, batch)
                    )

                torch.cuda.empty_cache()
                # memory_utils.print_memory(verbose=1)

            end_time = time.time() - start_time
            time_list.append(end_time)

        print(time_list)

    def perform_memory_test_vision_transformer(
        self,
        net,
        print_text="without compression",
    ):
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(
            net.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4
        )

        # if self.verbose:
        #     if self.depth is None:
        #         summary(net, (self.n_channels, self.width, self.height))

        print("=" * 60)
        print(print_text)
        print("=" * 60)

        time_list = list()
        for epoch in range(self.n_epoch):
            start_time = time.time()
            for batch in range(self.n_batch):
                if self.verbose:
                    print(
                        "Epoch {} --- Batch {} --- before forward".format(epoch, batch)
                    )

                torch.cuda.empty_cache()
                # memory_utils.print_memory(verbose=1)

                if self.frames is not None:
                    x = torch.randn(
                        self.batch_size,
                        self.n_channels,
                        self.frames,
                        self.width,
                        self.height,
                    ).cuda()
                else:
                    x = torch.randn(
                        self.batch_size,
                        self.n_channels,
                        self.width,
                        self.height,
                    ).cuda()

                target = (
                    torch.zeros(self.batch_size, self.n_class).type(torch.float).cuda()
                )
                y = net(x)

                if self.verbose:
                    print(
                        "Epoch {} --- Batch {} --- after forward".format(epoch, batch)
                    )

                torch.cuda.empty_cache()
                _, _, memory_allocated, _ = memory_utils.print_memory(verbose=1)

                # compute output
                loss = criterion(y, target)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.verbose:
                    print(
                        "Epoch {} --- Batch {} --- after backward".format(epoch, batch)
                    )

                torch.cuda.empty_cache()
                # memory_utils.print_memory(verbose=1)

            end_time = time.time() - start_time
            time_list.append(end_time)

        print(time_list)

        print()
        print()
        print(f"Mem. allocated: {memory_allocated:.0f}")


def main_mnist():
    M = MemoryTest(
        batch_size=2048,
        n_channels=1,
        width=28,
        height=28,
        depth=None,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    print("\n" * 3)

    M.perform_memory_test_classification(
        net=MNISTNet().cuda(), print_text="without compression"
    )

    print("\n" * 3)

    # M.perform_memory_test_classification(
    #     net=MNISTNetCompressed(
    #         compression_method=ThresholdingCompression,
    #         compression_parameters={
    #             "compression_ratio": 0.9,
    #         },
    #         # compressed_layers=["conv1", "relu1", "conv2", "maxpool"],
    #         compressed_layers=["conv1", "relu1", "conv2"],
    #         # compressed_layers=["conv1"],
    #         # compressed_layers=["relu1"],
    #         # compressed_layers=["conv2"],
    #         # compressed_layers=["maxpool"],
    #         # compressed_layers=["all"],
    #         # compressed_layers=[],
    #     ).cuda(),
    #     print_text="with Thresholding compression",
    # )

    # print("\n" * 3)

    # M.perform_memory_test_classification(
    #     net=MNISTNetCompressed(
    #         compression_method=DCTCompression,
    #         compression_parameters={
    #             "compression_ratio": 0.99999,
    #         },
    #         compressed_layers=["conv1", "relu1", "conv2", "maxpool"],
    #         # compressed_layers=["conv1", "relu1", "conv2"],
    #         # compressed_layers=["all"],
    #         # compressed_layers=[],
    #     ).cuda(),
    #     print_text="with DCT compression",
    # )

    # print("\n" * 3)

    # M.perform_memory_test_classification(
    #     net=MNISTNetCompressed(
    #         compression_method=WaveletTransformCompression,
    #         compression_parameters={
    #             "wave": "db3",
    #             "compression_ratio": 0.9,
    #             "n_levels": 3,
    #         },
    #         compressed_layers=["conv1", "relu1", "conv2", "maxpool"],
    #         # compressed_layers=["conv1", "relu1", "conv2"],
    #         # compressed_layers=["conv1"],
    #         # compressed_layers=["relu1"],
    #         # compressed_layers=["conv2"],
    #         # compressed_layers=["maxpool"],
    #         # compressed_layers=["all"],
    #         # compressed_layers=[],
    #     ).cuda(),
    #     print_text="with WT compression",
    # )

    print("\n" * 3)

    # M.perform_memory_test_classification(
    #     net=MNISTNetCompressed(
    #         compression_method=RandomCompression,
    #         compression_parameters=None,
    #         compressed_layers=["all"],
    #     ).cuda(),
    #     print_text="with random compression",
    # )


def main_lenet():
    """
    We are not using Lenet because using WT doesn't compress at all
    """

    M = MemoryTest(
        batch_size=8192,
        n_channels=1,
        width=28,
        height=28,
        depth=None,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    M.perform_memory_test_classification(
        net=LeNet5().cuda(), print_text="without compression"
    )
    M.perform_memory_test_classification(
        net=LeNet5Compressed(
            compression_method=WaveletTransformCompression,
            compression_parameters={
                "wave": "db3",
                "compression_ratio": 0.99,
                "n_levels": 5,
            },
            compressed_layers=["conv1", "relu1"],
            # compressed_layers=["conv1"],
        ).cuda(),
        print_text="with compression",
    )


def main_unet(memory_type):
    """
    We are not using Lenet because using WT doesn't compress at all
    NOTE: remember to set is_compressed_decoder=True
        in unet.py
    """

    M = MemoryTest(
        batch_size=1,
        n_channels=4,
        width=96,
        height=96,
        depth=96,
        n_class=3,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    M = MemoryTest(
        batch_size=2,
        n_channels=4,
        width=96,
        height=96,
        depth=96,
        n_class=3,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    M = MemoryTest(
        batch_size=1,
        n_channels=4,
        width=128,
        height=128,
        depth=128,
        n_class=3,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    # print("\n" * 3)

    if memory_type == "baseline":
        M.perform_memory_test_segmentation(
            net=UNet(in_channels=4, out_channels=3, n_base_filters=32).cuda(),
            print_text="without compression",
        )

    # print("\n" * 3)

    elif memory_type == "others":
        M.perform_memory_test_segmentation(
            net=UNetCompressed(
                in_channels=4,
                out_channels=3,
                n_base_filters=32,
                is_compressed=[True, True, True, True],
                compression_method=RandomCompression,
                compressed_layers=["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
            ).cuda(),
            print_text="with Random compression",
        )

    else:
        for is_compressed in [
            # [True, False, False, False],
            # [True, True, False, False],
            # [True, True, True, False],
            [True, True, True, True],
        ]:
            for compressed_layers in [
                # ["bn1", "bn2"],
                # ["conv1", "bn1", "conv2", "bn2"],
                ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
            ]:
                print()
                print()
                print()
                print()
                print(is_compressed)
                print(compressed_layers)

                # # WT + energy
                # M.perform_memory_test_segmentation(
                #     net=UNetCompressed(
                #         in_channels=4,
                #         out_channels=3,
                #         n_base_filters=32,
                #         is_compressed=is_compressed,
                #         compression_method=WaveletTransformCompression,
                #         compression_parameters={
                #             "wave": "db3",
                #             "compression_ratio": 0.9,
                #             "n_levels": 3,
                #         },
                #         compressed_layers=compressed_layers,
                #     ).cuda(),
                #     print_text="with WT compression",
                # )

                # DCT + topk
                M.perform_memory_test_segmentation(
                    net=UNetCompressed(
                        in_channels=4,
                        out_channels=3,
                        n_base_filters=32,
                        is_compressed=is_compressed,
                        compression_method=DCTCompression,
                        compression_parameters={
                            "compression_ratio": 0.99,
                        },
                        compressed_layers=compressed_layers,
                    ).cuda(),
                    print_text="with WT compression",
                )


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

    # M = MemoryTest(
    #     batch_size=512,
    #     n_channels=3,
    #     width=32,
    #     height=32,
    #     depth=None,
    #     n_class=100,
    #     n_epoch=1,
    #     n_batch=1,
    #     verbose=1,
    # )

    # M = MemoryTest(
    #     batch_size=128,
    #     n_channels=3,
    #     width=64,
    #     height=64,
    #     depth=None,
    #     n_class=100,
    #     n_epoch=1,
    #     n_batch=1,
    #     verbose=1,
    # )

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
            # [True, True, False, False, False],
            # [True, True, True, False, False],
            # [True, True, True, True, False],
            [True, True, True, True, True],
        ]:
            for compressed_layers_layer0, compressed_layers_layer1_4 in zip(
                [
                    ["bn1"],
                    # ["conv1", "bn1"],
                    # ["conv1", "bn1", "relu1"],
                ],
                [
                    ["bn1", "bn2"],
                    # ["conv1", "bn1", "conv2", "bn2"],
                    # ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
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


def main_resnet152_basic(memory_type):
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
            net=ResNet(BasicBlock, [3, 8, 36, 3], num_classes=100).cuda(),
            print_text="without compression",
        )

    ## Mem. Others
    elif memory_type == "others":
        M.perform_memory_test_classification(
            net=ResNetCompressed(
                num_classes=100,
                num_blocks=[3, 8, 36, 3],
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
            # [True, True, False, False, False],
            [True, True, True, False, False],
            # [True, True, True, True, False],
            # [True, True, True, True, True],
        ]:
            for compressed_layers_layer0, compressed_layers_layer1_4 in zip(
                [
                    # ["bn1"],
                    # ["conv1", "bn1"],
                    ["conv1", "bn1", "relu1"],
                ],
                [
                    # ["bn1", "bn2"],
                    # ["conv1", "bn1", "conv2", "bn2"],
                    ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
                ],
            ):
                print()
                print()
                print()
                print()
                print(is_compressed)
                print(compressed_layers_layer0)
                print(compressed_layers_layer1_4)

                M.perform_memory_test_classification(
                    net=ResNetCompressed(
                        num_classes=100,
                        num_blocks=[3, 8, 36, 3],
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


def main_resnet152_bottleneck(memory_type):
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
            net=ResNet(Bottleneck, [3, 8, 36, 3], num_classes=100).cuda(),
            print_text="without compression",
        )

    ## Mem. Others
    elif memory_type == "others":
        M.perform_memory_test_classification(
            net=ResNetCompressed(
                num_classes=100,
                num_blocks=[3, 8, 36, 3],
                block=Bottleneck,
                block_compressed=BottleneckCompressed,
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
                    "conv3",
                    "bn3",
                    "relu3",
                ],
            ).cuda(),
            print_text="without compression",
        )

    ## Mem.
    else:
        for is_compressed in [
            # [True, False, False, False, False],
            # [True, True, False, False, False],
            # [True, True, True, False, False],
            # [True, True, True, True, False],
            [True, True, True, True, True],
        ]:
            for compressed_layers_layer0, compressed_layers_layer1_4 in zip(
                [
                    # ["bn1"],
                    # ["conv1", "bn1"],
                    ["conv1", "bn1", "relu1"],
                ],
                [
                    # ["bn1", "bn2", "bn3"],
                    # ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3"],
                    [
                        "conv1",
                        "bn1",
                        "relu1",
                        "conv2",
                        "bn2",
                        "relu2",
                        "conv3",
                        "bn3",
                        "relu3",
                    ],
                ],
            ):
                print()
                print()
                print()
                print()
                print(is_compressed)
                print(compressed_layers_layer0)
                print(compressed_layers_layer1_4)

                M.perform_memory_test_classification(
                    net=ResNetCompressed(
                        num_classes=100,
                        num_blocks=[3, 8, 36, 3],
                        block=Bottleneck,
                        block_compressed=BottleneckCompressed,
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


def main_wideresnet101_bottleneck(memory_type):
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
            net=ResNet(
                Bottleneck,
                [3, 4, 23, 3],
                num_classes=100,
                width_per_group=128,
            ).cuda(),
            print_text="without compression",
        )

    ## Mem. Others
    elif memory_type == "others":
        M.perform_memory_test_classification(
            net=ResNetCompressed(
                num_classes=100,
                num_blocks=[3, 4, 23, 3],
                width_per_group=128,
                block=Bottleneck,
                block_compressed=BottleneckCompressed,
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
                    "conv3",
                    "bn3",
                    "relu3",
                ],
            ).cuda(),
            print_text="without compression",
        )

    ## Mem.
    else:
        for is_compressed in [
            # [True, False, False, False, False],
            # [True, True, False, False, False],
            # [True, True, True, False, False],
            # [True, True, True, True, False],
            [True, True, True, True, True],
        ]:
            for compressed_layers_layer0, compressed_layers_layer1_4 in zip(
                [
                    # ["bn1"],
                    # ["conv1", "bn1"],
                    ["conv1", "bn1", "relu1"],
                ],
                [
                    # ["bn1", "bn2", "bn3"],
                    # ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3"],
                    [
                        "conv1",
                        "bn1",
                        "relu1",
                        "conv2",
                        "bn2",
                        "relu2",
                        "conv3",
                        "bn3",
                        "relu3",
                    ],
                ],
            ):
                print()
                print()
                print()
                print()
                print(is_compressed)
                print(compressed_layers_layer0)
                print(compressed_layers_layer1_4)

                M.perform_memory_test_classification(
                    net=ResNetCompressed(
                        num_classes=100,
                        num_blocks=[3, 4, 23, 3],
                        width_per_group=128,
                        block=Bottleneck,
                        block_compressed=BottleneckCompressed,
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


def main_vit3d(memory_type):
    """
    We are not using Lenet because using WT doesn't compress at all
    """

    image_size = 128
    batch_size = 32
    depth = 5

    M = MemoryTest(
        batch_size=batch_size,
        n_channels=3,
        width=image_size,
        height=image_size,
        frames=16,
        depth=depth,
        n_class=100,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    ## Mem. Baseline
    if memory_type == "baseline":
        M.perform_memory_test_vision_transformer(
            net=ViT3d(
                image_size=image_size,  # image size
                frames=16,  # number of frames
                image_patch_size=16,  # image patch size
                frame_patch_size=2,  # frame patch size
                num_classes=100,
                dim=1024,
                depth=depth,
                heads=8,
                mlp_dim=2048,
                dropout=0,
                emb_dropout=0,
            ).cuda(),
            print_text="without compression",
        )

    ## Mem. Others
    elif memory_type == "others":
        M.perform_memory_test_vision_transformer(
            net=ViTCompressed3d(
                image_size=image_size,  # image size
                frames=16,  # number of frames
                image_patch_size=16,  # image patch size
                frame_patch_size=2,  # frame patch size
                num_classes=100,
                dim=1024,
                depth=depth,
                heads=8,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1,
            ).cuda(),
            print_text="without compression",
        )

    ## Mem.
    else:
        for is_compressed in [
            [True, False, False, False, False],
            # [True, True, False, False, False],
            # [True, True, True, False, False],
            # [True, True, True, True, False],
            # [True, True, True, True, True],
        ]:
            M.perform_memory_test_vision_transformer(
                net=ViTCompressed3d(
                    image_size=image_size,  # image size
                    frames=16,  # number of frames
                    image_patch_size=16,  # image patch size
                    frame_patch_size=2,  # frame patch size
                    num_classes=100,
                    dim=1024,
                    depth=depth,
                    heads=8,
                    mlp_dim=2048,
                    dropout=0.1,
                    emb_dropout=0.1,
                    compression_method=DCTCompression,
                    compression_parameters={
                        "compression_ratio": 0.99,
                    },
                    is_compressed=is_compressed,
                ).cuda(),
                print_text="without compression",
            )


def main_vit(memory_type):
    """
    We are not using Lenet because using WT doesn't compress at all
    """

    image_size = 256

    M = MemoryTest(
        batch_size=256,
        n_channels=3,
        width=image_size,
        height=image_size,
        # frames=16,
        depth=None,
        n_class=1000,
        n_epoch=1,
        n_batch=1,
        verbose=1,
    )

    ## Mem. Baseline
    if memory_type == "baseline":
        M.perform_memory_test_vision_transformer(
            net=ViT(
                image_size=image_size,
                patch_size=32,
                num_classes=1000,
                dim=1024,
                depth=5,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1,
            ).cuda(),
            print_text="without compression",
        )

    ## Mem. Others
    elif memory_type == "others":
        M.perform_memory_test_vision_transformer(
            net=ViTCompressed(
                image_size=image_size,
                patch_size=32,
                num_classes=1000,
                dim=1024,
                depth=5,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1,
                compression_method=RandomCompression,
                compression_parameters=None,
                is_compressed=[True, True, True, True, True],
            ).cuda(),
            print_text="without compression",
        )

    ## Mem.
    else:
        for is_compressed in [
            # [True, False, False, False, False],
            # [True, True, False, False, False],
            # [True, True, True, False, False],
            # [True, True, True, True, False],
            [True, True, True, True, True],
        ]:
            M.perform_memory_test_vision_transformer(
                net=ViTCompressed(
                    image_size=image_size,
                    patch_size=32,
                    num_classes=1000,
                    dim=1024,
                    depth=5,
                    heads=16,
                    mlp_dim=2048,
                    dropout=0.1,
                    emb_dropout=0.1,
                    compression_method=DCTCompression,
                    compression_parameters={
                        "compression_ratio": 0.99,
                    },
                    is_compressed=is_compressed,
                ).cuda(),
                print_text="without compression",
            )
