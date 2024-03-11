"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import engine.wavelet as wavelet
import engine.utils.memory_utils as memory_utils

from engine.compression import (
    WaveletTransformCompression,
    DCTCompression,
)


BS = 1024
Channel = 3
W, H, D = 32, 32, 32


class BasicBlockCompressedRand(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockCompressedRand, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def pack_hook(self, tensor):
        # print("Packing")
        return tensor.size()

    def unpack_hook(self, size):
        # print("Unpacking")
        return torch.rand(size).to("cuda")

    def forward(self, x):
        with torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook):
            out = self.conv1(x)
            out = F.relu(self.bn1(out))
            # with torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook):
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups: int = 1,
        base_width: int = 64,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlockCompressed(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups: int = 1,
        base_width: int = 64,
        compression_method=WaveletTransformCompression,
        compression_parameters=None,
        compressed_layers=["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
    ):
        super(BasicBlockCompressed, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

        self.compression = compression_method(self, compression_parameters)
        self.compressed_layers = compressed_layers

    def forward(self, x):
        x0 = x
        for layer in ["conv1", "bn1", "relu1", "conv2", "bn2"]:
            if layer in self.compressed_layers or "all" in self.compressed_layers:
                with torch.autograd.graph.saved_tensors_hooks(
                    self.compression.pack_hook, self.compression.unpack_hook
                ):
                    x = getattr(self, layer)(x)
            else:
                x = getattr(self, layer)(x)

        x += self.shortcut(x0)

        for layer in ["relu2"]:
            if layer in self.compressed_layers or "all" in self.compressed_layers:
                with torch.autograd.graph.saved_tensors_hooks(
                    self.compression.pack_hook, self.compression.unpack_hook
                ):
                    x = getattr(self, layer)(x)
            else:
                x = getattr(self, layer)(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups: int = 1,
        base_width: int = 64,
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckCompressed(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups: int = 1,
        base_width: int = 64,
        compression_method=WaveletTransformCompression,
        compression_parameters=None,
        compressed_layers=[
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
    ):
        super(BottleneckCompressed, self).__init__()

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

        self.compression = compression_method(self, compression_parameters)
        self.compressed_layers = compressed_layers

    def forward(self, x):
        x0 = x

        for layer in ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2", "conv3", "bn3"]:
            if layer in self.compressed_layers or "all" in self.compressed_layers:
                with torch.autograd.graph.saved_tensors_hooks(
                    self.compression.pack_hook, self.compression.unpack_hook
                ):
                    x = getattr(self, layer)(x)
            else:
                x = getattr(self, layer)(x)

        x += self.shortcut(x0)

        for layer in ["relu3"]:
            if layer in self.compressed_layers or "all" in self.compressed_layers:
                with torch.autograd.graph.saved_tensors_hooks(
                    self.compression.pack_hook, self.compression.unpack_hook
                ):
                    x = getattr(self, layer)(x)
            else:
                x = getattr(self, layer)(x)

        return x


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        input_dim: list[int] = [32, 32],
        in_planes=64,
        num_classes=10,
        groups: int = 1,
        width_per_group: int = 64,
    ):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.groups = groups
        self.base_width = width_per_group
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.dim_linear = int(self.input_dim[0] * self.input_dim[1] / 2)
        self.linear = nn.Linear(self.dim_linear * block.expansion, num_classes)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)
        # self.linear = nn.Linear(2048 * block.expansion, num_classes) # for 3x64x64

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, self.groups, self.base_width)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetCompressed(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        block_compressed=BasicBlockCompressed,
        num_blocks=[2, 2, 2, 2],
        num_classes=10,
        in_planes=64,
        groups: int = 1,
        width_per_group: int = 64,
        is_compressed=[
            True,
            False,
            False,
            False,
            False,
        ],  # layer0, layer1, layer2, layer3, layer4
        compression_method=WaveletTransformCompression,
        compression_parameters={
            "wave": "db3",
            "compression_ratio": 0.9,
            "n_levels": 3,
        },
        compressed_layers_layer0=["conv1", "bn1", "relu1"],
        compressed_layers_layer1_4=["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
    ):
        super(ResNetCompressed, self).__init__()

        self.in_planes = in_planes
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # self.linear = nn.Linear(2048 * block.expansion, num_classes) # for 3x64x64

        self.is_compressed = is_compressed
        self.compression_method = compression_method
        self.compression_parameters = compression_parameters
        self.compression = compression_method(self, compression_parameters)
        self.compressed_layers_layer0 = compressed_layers_layer0
        self.compressed_layers_layer1_4 = compressed_layers_layer1_4

        if self.is_compressed[1]:
            self.layer1 = self._make_layer_compressed(
                block_compressed,
                64,
                num_blocks[0],
                stride=1,
                compression_method=self.compression_method,
                compression_parameters=self.compression_parameters,
                compressed_layers=self.compressed_layers_layer1_4,
            )
        else:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

        if self.is_compressed[2]:
            self.layer2 = self._make_layer_compressed(
                block_compressed,
                128,
                num_blocks[1],
                stride=2,
                compression_method=self.compression_method,
                compression_parameters=self.compression_parameters,
                compressed_layers=self.compressed_layers_layer1_4,
            )
        else:
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        if self.is_compressed[3]:
            self.layer3 = self._make_layer_compressed(
                block_compressed,
                256,
                num_blocks[2],
                stride=2,
                compression_method=self.compression_method,
                compression_parameters=self.compression_parameters,
                compressed_layers=self.compressed_layers_layer1_4,
            )
        else:
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        if self.is_compressed[4]:
            self.layer4 = self._make_layer_compressed(
                block_compressed,
                512,
                num_blocks[3],
                stride=2,
                compression_method=self.compression_method,
                compression_parameters=self.compression_parameters,
                compressed_layers=self.compressed_layers_layer1_4,
            )
        else:
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_compressed(
        self,
        block,
        planes,
        num_blocks,
        stride,
        compression_method,
        compression_parameters,
        compressed_layers,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    self.groups,
                    self.base_width,
                    compression_method,
                    compression_parameters,
                    compressed_layers,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.is_compressed[0]:
            out = x
            for layer in ["conv1", "bn1", "relu1"]:
                if (
                    layer in self.compressed_layers_layer0
                    or "all" in self.compressed_layers_layer0
                ):
                    with torch.autograd.graph.saved_tensors_hooks(
                        self.compression.pack_hook, self.compression.unpack_hook
                    ):
                        out = getattr(self, layer)(out)
                else:
                    out = getattr(self, layer)(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet18Compressed_rand():
    return ResNetCompressed(
        [
            BasicBlockCompressedRand,
            BasicBlockCompressedRand,
            BasicBlockCompressedRand,
            BasicBlockCompressedRand,
        ],
        [2, 2, 2, 2],
    )


def ResNet18Compressed():
    return ResNetCompressed(
        [
            BasicBlockCompressed,
            BasicBlock,
            BasicBlock,
            BasicBlock,
            # BasicBlockCompressed,
            # BasicBlockCompressed,
            # BasicBlockCompressed,
        ],
        [2, 2, 2, 2],
    )


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def WideResNet101():
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
    )


def ResNet152Compressed():
    return ResNetCompressed(
        [
            BasicBlockCompressed,
            BasicBlock,
            BasicBlock,
            BasicBlock,
            # BasicBlockCompressed,
            # BasicBlockCompressed,
            # BasicBlockCompressed,
        ],
        [3, 8, 36, 3],
    )


def WideResNet101Compressed():
    return ResNetCompressed(
        [
            BasicBlockCompressed,
            BasicBlock,
            BasicBlock,
            BasicBlock,
            # BasicBlockCompressed,
            # BasicBlockCompressed,
            # BasicBlockCompressed,
        ],
        [3, 4, 23, 3],
        in_planes=128,
    )


def without_compression(n_epoch, n_batch, verbose=1):
    # no compression
    net = ResNet18().cuda()

    summary(net, (3, 32, 32))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
    print("=" * 60)
    print("without compression")
    print("=" * 60)

    time_list = list()
    for epoch in range(n_epoch):
        start_time = time.time()
        for batch in range(n_batch):
            if verbose:
                print("Epoch {} --- Batch {} --- before forward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

            x = torch.randn(
                BS,
                3,
                W,
                H,
            ).cuda()

            target = torch.zeros(BS).type(torch.LongTensor).cuda()

            y = net(x)

            if verbose:
                print("Epoch {} --- Batch {} --- after forward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

            # compute output
            loss = criterion(y, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                print("Epoch {} --- Batch {} --- after backward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

        end_time = time.time() - start_time
        time_list.append(end_time)

    print(time_list)


def with_compression_hook(n_epoch, n_batch, verbose=1):
    # with compression
    # no compression

    net = ResNetCompressed(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        block_compressed=BasicBlockCompressed,
        # is_compressed=[
        #     False,
        #     True,
        #     True,
        #     True,
        #     True,
        # ],  # layer0, layer1, layer2, layer3, layer4
        is_compressed=[
            False,
            True,
            False,
            False,
            False,
        ],  # layer0, layer1, layer2, layer3, layer4
        compression_method=WaveletTransformCompression,
        compression_parameters={
            "wave": "db3",
            "compression_ratio": 0.9,
            "n_levels": 3,
        },
        # compressed_layers_layer0=["conv1", "bn1", "relu1"],
        # compressed_layers_layer0=["bn1", "relu1"],
        compressed_layers_layer0=[
            "bn1",
        ],
        compressed_layers_layer1_4=["bn1", "bn2"],
        # compressed_layers_layer1_4=["bn1"],
        # compressed_layers_layer1_4=["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
        # compressed_layers_layer1_4=["conv1", "conv2"],
        # compressed_layers_layer1_4=["relu1"],
    ).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
    print("\n" * 2)
    print("=" * 60)
    print("with compression")
    print("=" * 60)

    time_list = list()
    for epoch in range(n_epoch):
        start_time = time.time()
        for batch in range(n_batch):
            if verbose:
                print("Epoch {} --- Batch {} --- before forward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

            x = torch.randn(BS, 3, W, H).cuda()

            target = torch.zeros(BS).type(torch.LongTensor).cuda()

            y = net(x)

            if verbose:
                print("Epoch {} --- Batch {} --- after forward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

            # compute output
            loss = criterion(y, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                print("Epoch {} --- Batch {} --- after backward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

        end_time = time.time() - start_time
        time_list.append(end_time)

    print(time_list)


def with_compression_hook_rand(n_epoch, n_batch, verbose=1):
    # with compression
    # no compression
    net = ResNet18Compressed_rand().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
    print("\n" * 2)
    print("=" * 60)
    print("with rand")
    print("=" * 60)

    time_list = list()
    for epoch in range(n_epoch):
        start_time = time.time()
        for batch in range(n_batch):
            if verbose:
                print("Epoch {} --- Batch {} --- before forward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

            x = torch.randn(BS, 3, W, H).cuda()

            target = torch.zeros(BS).type(torch.LongTensor).cuda()

            y = net(x)

            if verbose:
                print("Epoch {} --- Batch {} --- after forward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

            # compute output
            loss = criterion(y, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                print("Epoch {} --- Batch {} --- after backward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

        end_time = time.time() - start_time
        time_list.append(end_time)

    print(time_list)


def test(n_epoch, n_batch, verbose=1):
    net = ResNetCompressed(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        block_compressed=BasicBlockCompressed,
        is_compressed=[
            True,
            False,
            False,
            False,
            False,
        ],  # layer0, layer1, layer2, layer3, layer4
        compression_method=WaveletTransformCompression,
        compression_parameters={
            "wave": "db3",
            "compression_ratio": 0.9,
            "n_levels": 3,
        },
        compressed_layers_layer0=["conv1", "bn1", "relu1"],
        compressed_layers_layer1_4=["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
    ).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
    print("\n" * 2)
    print("=" * 60)
    print("with rand")
    print("=" * 60)

    time_list = list()
    for epoch in range(n_epoch):
        start_time = time.time()
        for batch in range(n_batch):
            if verbose:
                print("Epoch {} --- Batch {} --- before forward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

            x = torch.randn(BS, 3, W, H).cuda()

            target = torch.zeros(BS).type(torch.LongTensor).cuda()

            y = net(x)

            if verbose:
                print("Epoch {} --- Batch {} --- after forward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

            # compute output
            loss = criterion(y, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                print("Epoch {} --- Batch {} --- after backward".format(epoch, batch))

            torch.cuda.empty_cache()
            memory_utils.print_memory(verbose=1)

        end_time = time.time() - start_time
        time_list.append(end_time)

    print(time_list)


def compare():
    x = torch.randn(1, 3, W, H).cuda()

    net_compressed = ResNetCompressed(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        block_compressed=BasicBlockCompressed,
        is_compressed=[
            True,
            True,
            False,
            False,
            False,
        ],
        compression_method=DCTCompression,
        compression_parameters={
            "wave": "db3",
            "compression_ratio": 0.1,
            "n_levels": 3,
        },
        compressed_layers_layer0=["bn1"],
        compressed_layers_layer1_4=[
            "bn1",
            "bn2",
        ],
    ).cuda()

    net = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()

    net.load_state_dict(net_compressed.state_dict())

    y_compressed = net_compressed(x)

    y = net(x)

    print(torch.all(y.eq(y_compressed)))

    a = 2


def main():
    # with_compression()
    verbose = 1
    n_epoch = 1
    n_batch = 1

    without_compression(n_epoch, n_batch, verbose)
    with_compression_hook(n_epoch, n_batch, verbose)
    # with_compression_hook_rand(n_epoch, n_batch, verbose)

    # test(n_epoch, n_batch, verbose)


if __name__ == "__main__":
    # main()
    compare()


# test()


"""

@TODO
- generate general class to get memory for all tasks (classification + segmentation)
- run unet



class Compression():
    def pack_hook():
        pass
    def unpack_hook():
        pass


class MyNetworkBlock():
    pass  # Define the network here


class CompressedBlock():
    def __init__(self, network_block, compression):
        self.network_block = network_block
        self.compression = compression

    def forward(self, x):
        with torch.autograd.graph.saved_tensors_hooks(self.compression.pack_hook, self.compression.unpack_hook):
            out = self.network_block(x)
        return out

"""


"""

Resnet18

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------


----------------------------------------------------------------
Conclusion:
- WT affects compression of tensor with small W x H
- Compression techniques matter
- Interchangable class of compression
- Thresholding -> sparse
----------------------------------------------------------------



============================================================
without compression
============================================================
------------------------------------------------------------
Epoch 0 --- Batch 0 --- after forward
------------------------------------------------------------
max_memory_allocated: 4778.72MB
max_memory_cached: 4966.00MB
memory_allocated: 4632.76MB
memory_cached: 4784.00MB


============================================================
with compression - 1 block
============================================================
------------------------------------------------------------
Epoch 0 --- Batch 0 --- after forward
------------------------------------------------------------
max_memory_allocated: 6242.10MB
max_memory_cached: 8000.00MB
memory_allocated: 4411.33MB
memory_cached: 4614.00MB


============================================================
with compression - all blocks
============================================================
------------------------------------------------------------
Epoch 0 --- Batch 0 --- after forward
------------------------------------------------------------
max_memory_allocated: 7495.49MB
max_memory_cached: 8000.00MB
memory_allocated: 6712.65MB
memory_cached: 7632.00MB


============================================================
with rand
============================================================
------------------------------------------------------------
Epoch 0 --- Batch 0 --- after forward
------------------------------------------------------------
max_memory_allocated: 6242.10MB
max_memory_cached: 8000.00MB
memory_allocated: 600.73MB
memory_cached: 848.00MB

"""
