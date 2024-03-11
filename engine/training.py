"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import os
import sys
import time
import shutil
import json
import glob

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import engine.logger as logger
import engine.utils.memory_utils as memory_utils

from monai.utils import first, set_determinism
from monai.apps import DecathlonDataset
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
)

from models.clasification import *
from models.segmentation import *

import engine.compression as compression


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = shutil.get_terminal_size()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def build_model(options):
    model_name = options["model"]["arch"]
    dataset = options["data"]["dataset"]
    if dataset == "mnist":
        if model_name == "lenet5":
            net = LeNet5()
        elif model_name == "mnistnet":
            net = MNISTNet()
        elif model_name == "mnistnet_freeze":
            net = MNISTNet()
            for name, param in net.named_parameters():
                if param.requires_grad and ("conv1" in name) or ("conv2" in name):
                    param.requires_grad = False
        elif model_name == "mnistnet_compressed":
            (
                compression_method,
                compression_parameters,
                compressed_layers,
            ) = compression.get_compression_params(options)
            net = MNISTNetCompressed(
                compression_method=compression_method,
                compression_parameters=compression_parameters,
                compressed_layers=compressed_layers,
            )
        else:
            raise ValueError("Model is NotImplemented. Please check")

    elif dataset == "cifar10":
        if model_name == "resnet18":
            net = ResNet18()
        else:
            raise ValueError("Model is NotImplemented. Please check")

    elif dataset == "cifar100":
        if model_name == "resnet18":
            net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)
        elif model_name == "resnet18_compressed":
            (
                compression_method,
                compression_parameters,
                compressed_blocks,
            ) = compression.get_compression_params(options)

            is_compressed = []
            for b in ["block0", "block1", "block2", "block3", "block4"]:
                if b in compressed_blocks:
                    is_compressed.append(True)
                else:
                    is_compressed.append(False)

            net = ResNetCompressed(
                num_classes=100,
                is_compressed=is_compressed,
                compression_method=compression_method,
                compression_parameters=compression_parameters,
                compressed_layers_layer0=["bn1"],
                compressed_layers_layer1_4=[
                    "bn1",
                    "bn2",
                ],
                # compressed_layers_layer0=["conv1", "bn1", "relu1"],
                # compressed_layers_layer1_4=[
                #     "conv1",
                #     "bn1",
                #     "relu1",
                #     "conv2",
                #     "bn2",
                #     "relu2",
                # ],
            )

        elif model_name == "resnet18_compressed_regu":
            (
                compression_method,
                compression_parameters,
                compressed_layers,
            ) = compression.get_compression_params(options)

            is_compressed = []

            # for regularing effect experiment, we compress only block0
            for b in ["block0", "block1", "block2", "block3", "block4"]:
                if b == "block0":
                    is_compressed.append(True)
                else:
                    is_compressed.append(False)

            net = ResNetCompressed(
                num_classes=100,
                is_compressed=is_compressed,
                compression_method=compression_method,
                compression_parameters=compression_parameters,
                compressed_layers_layer0=compressed_layers,
                compressed_layers_layer1_4=[],
            )

    elif dataset == "brats":
        if model_name == "unet":
            net = UNet(in_channels=4, out_channels=3, n_base_filters=32)
        elif model_name == "unet_compressed":
            (
                compression_method,
                compression_parameters,
                compressed_blocks,
            ) = compression.get_compression_params(options)

            is_compressed = []
            for b in ["block0", "block1", "block2", "block3", "block4"]:
                if b in compressed_blocks:
                    is_compressed.append(True)
                else:
                    is_compressed.append(False)

            net = UNetCompressed(
                in_channels=4,
                out_channels=3,
                n_base_filters=32,
                is_compressed=is_compressed,
                compression_method=compression_method,
                compression_parameters=compression_parameters,
                compressed_layers=[
                    "bn1",
                    "bn2",
                ],
                # compressed_layers=[
                #     "conv1",
                #     "bn1",
                #     "relu1",
                #     "conv2",
                #     "bn2",
                #     "relu2",
                # ],
            )
        elif model_name == "unet_compressed_regu":
            (
                compression_method,
                compression_parameters,
                compressed_layers,
            ) = compression.get_compression_params(options)

            is_compressed = []
            for b in ["block0", "block1", "block2", "block3", "block4"]:
                if b == "block0":
                    is_compressed.append(True)
                else:
                    is_compressed.append(False)

            net = UNetCompressed(
                in_channels=4,
                out_channels=3,
                n_base_filters=32,
                is_compressed=is_compressed,
                compression_method=compression_method,
                compression_parameters=compression_parameters,
                compressed_layers=compressed_layers,
            )

    elif dataset == "spleen":
        if model_name == "unet":
            net = UNet(in_channels=1, out_channels=2, n_base_filters=14)
        elif model_name == "unet_compressed":
            (
                compression_method,
                compression_parameters,
                compressed_blocks,
            ) = compression.get_compression_params(options)

            is_compressed = []
            for b in ["block0", "block1", "block2", "block3", "block4"]:
                if b in compressed_blocks:
                    is_compressed.append(True)
                else:
                    is_compressed.append(False)

            net = UNetCompressed(
                in_channels=1,
                out_channels=2,
                n_base_filters=14,
                is_compressed=is_compressed,
                compression_method=compression_method,
                compression_parameters=compression_parameters,
                compressed_layers=[
                    "bn1",
                    "bn2",
                ],
            )
        elif model_name == "unet_compressed_regu":
            (
                compression_method,
                compression_parameters,
                compressed_layers,
            ) = compression.get_compression_params(options)

            is_compressed = []
            for b in ["block0", "block1", "block2", "block3", "block4"]:
                if b == "block0":
                    is_compressed.append(True)
                else:
                    is_compressed.append(False)

            net = UNetCompressed(
                in_channels=1,
                out_channels=2,
                n_base_filters=14,
                is_compressed=is_compressed,
                compression_method=compression_method,
                compression_parameters=compression_parameters,
                compressed_layers=compressed_layers,
            )

        else:
            raise ValueError("Model is NotImplemented. Please check")

    else:
        raise ValueError("Dataset is NotSupported. Please check")
    return net


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1)
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


def setup_dataloader_transform(args, options):
    def get_subset(
        args, options, dataset, is_test=False, percent=10
    ):  # get subset of data for testing
        if is_test:
            is_shuffle = False
        else:
            is_shuffle = True

        dataset_name = options["data"]["dataset"]
        if dataset_name in ["mnist", "cifar100"]:
            num_workers = 0
        elif dataset_name == "brats":
            num_workers = args.workers
        elif dataset_name == "spleen":
            num_workers = 4

        if args.is_exp_epoch_time:
            subset_dataset = torch.utils.data.random_split(
                dataset,
                [
                    int(len(dataset) * percent / 100),
                    len(dataset) - int(len(dataset) * percent / 100),
                ],
            )[0]
            loader = DataLoader(
                subset_dataset,
                batch_size=options["optim"]["batch_size"],
                shuffle=is_shuffle,
                num_workers=num_workers,
            )
        else:
            if dataset_name == "spleen":
                subset_dataset = torch.utils.data.random_split(
                    dataset,
                    [
                        int(len(dataset)),
                        len(dataset) - int(len(dataset)),
                    ],
                )[0]
                loader = DataLoader(
                    subset_dataset,
                    batch_size=options["optim"]["batch_size"],
                    shuffle=is_shuffle,
                    num_workers=num_workers,
                )
            else:
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=options["optim"]["batch_size"],
                    shuffle=is_shuffle,
                    num_workers=num_workers,
                )

        return loader

    dataset = options["data"]["dataset"]

    if dataset == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_dataset = torchvision.datasets.MNIST(
            root=options["data"]["dir"],
            train=True,
            download=True,
            transform=transform_train,
        )

        trainloader = get_subset(args, options, train_dataset)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset
            )
        else:
            train_sampler = None

        test_dataset = torchvision.datasets.MNIST(
            root=options["data"]["dir"],
            train=False,
            download=True,
            transform=transform_test,
        )

        testloader = get_subset(args, options, test_dataset, is_test=True)

        classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    elif dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root=options["data"]["dir"],
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = get_subset(args, options, train_dataset)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset
            )
        else:
            train_sampler = None

        test_dataset = torchvision.datasets.CIFAR10(
            root=options["data"]["dir"],
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = get_subset(args, options, test_dataset, is_test=True)

        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    elif dataset == "cifar100":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR100(
            root=options["data"]["dir"],
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = get_subset(args, options, train_dataset)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset
            )
        else:
            train_sampler = None

        test_dataset = torchvision.datasets.CIFAR100(
            root=options["data"]["dir"],
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = get_subset(args, options, test_dataset, is_test=True)

    elif dataset == "imagenet":
        traindir = os.path.join(options["data"]["dir"], "train")
        testdir = os.path.join(options["data"]["dir"], "val")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        test_dataset = datasets.ImageFolder(
            testdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset
            )
        else:
            train_sampler = None

        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=options["optim"]["batch_size"],
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
        )

        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=options["optim"]["batch_size"],
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    elif dataset == "brats":
        train_transform_old = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=[224, 224, 144],
                    random_size=False,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        val_transform_old = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        train_transform = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Resized(
                    keys=["image", "label"],
                    # spatial_size=(128, 128, 128),
                    spatial_size=(96, 96, 96),
                    size_mode="all",
                    mode=("trilinear", "nearest"),
                ),
                RandSpatialCropd(
                    keys=["image", "label"],
                    # roi_size=[112, 112, 72],
                    # roi_size=[224, 224, 144],
                    # roi_size=[128, 128, 128],
                    roi_size=[96, 96, 96],
                    random_size=False,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Resized(
                    keys=["image", "label"],
                    # spatial_size=(128, 128, 128),
                    spatial_size=(96, 96, 96),
                    size_mode="all",
                    mode=("trilinear", "nearest"),
                ),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        train_dataset = DecathlonDataset(
            root_dir=options["data"]["dir"],  # root_dir
            task="Task01_BrainTumour",
            transform=train_transform,
            section="training",
            download=False,
            cache_rate=0.0,
            num_workers=args.workers,
        )
        trainloader = get_subset(args, options, train_dataset)
        val_dataset = DecathlonDataset(
            root_dir=options["data"]["dir"],
            task="Task01_BrainTumour",
            transform=val_transform,
            section="validation",
            download=False,
            cache_rate=0.0,
            num_workers=args.workers,
        )
        valloader = get_subset(args, options, val_dataset, is_test=True)
        testloader = valloader

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset
            )
        else:
            train_sampler = None

    elif dataset == "spleen":
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        train_dataset = DecathlonDataset(
            root_dir=options["data"]["dir"],  # root_dir
            task="Task09_Spleen",
            transform=train_transforms,
            section="training",
            download=False,
            cache_rate=0.0,
            num_workers=args.workers,
        )
        trainloader = get_subset(args, options, train_dataset)
        val_dataset = DecathlonDataset(
            root_dir=options["data"]["dir"],
            task="Task09_Spleen",
            transform=val_transforms,
            section="validation",
            download=False,
            cache_rate=0.0,
            num_workers=args.workers,
        )
        valloader = get_subset(args, options, val_dataset, is_test=True)
        testloader = valloader

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset
            )
        else:
            train_sampler = None

    elif dataset == "spleen_old":
        set_determinism(seed=0)

        train_images = sorted(
            glob.glob(
                os.path.join(
                    options["data"]["dir"], "Task09_Spleen", "imagesTr", "*.nii.gz"
                )
            )
        )
        train_labels = sorted(
            glob.glob(
                os.path.join(
                    options["data"]["dir"], "Task09_Spleen", "labelsTr", "*.nii.gz"
                )
            )
        )
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]
        train_files, val_files = data_dicts[:-9], data_dicts[-9:]

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                # user can also add other random transforms
                # RandAffined(
                #     keys=['image', 'label'],
                #     mode=('bilinear', 'nearest'),
                #     prob=1.0, spatial_size=(96, 96, 96),
                #     rotate_range=(0, 0, np.pi/15),
                #     scale_range=(0.1, 0.1, 0.1)),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        """## Check transforms in DataLoader"""
        check_ds = Dataset(data=val_files, transform=val_transforms)
        check_loader = DataLoader(check_ds, batch_size=1)
        check_data = first(check_loader)
        image, label = (check_data["image"][0][0], check_data["label"][0][0])
        print(f"image shape: {image.shape}, label shape: {label.shape}")

        train_ds = CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4
        )
        trainloader = get_subset(args, options, train_ds)

        val_ds = CacheDataset(
            data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4
        )
        testloader = get_subset(args, options, val_ds, is_test=True)

        train_sampler = None

    else:
        raise ValueError("Dataset is NotSupported. Please check")

    return trainloader, testloader, train_sampler


def setup_criterion_optimizer(model, args, options):
    dataset = options["data"]["dataset"]

    if dataset in ["mnist", "cifar10", "cifar100", "imagenet"]:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=options["optim"]["learning_rate"], betas=(0.9, 0.999)
        )

        return criterion, optimizer

    elif dataset == "brats":
        from monai.losses import DiceLoss

        criterion = DiceLoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        dice_metric = DiceMetric(include_background=True, reduction="mean")
        dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

        return criterion, optimizer, lr_scheduler, dice_metric, dice_metric_batch

    elif dataset == "spleen":
        from monai.losses import DiceLoss

        criterion = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        dice_metric = DiceMetric(include_background=False, reduction="mean")

        return criterion, optimizer, dice_metric


def save_results(results, epoch, split_name, dir_logs, dir_vqa):
    dir_epoch = os.path.join(dir_logs, "epoch_" + str(epoch))
    name_json = "OpenEnded_mscoco_{}_model_results.json".format(split_name)
    # TODO: simplify formating
    if "test" in split_name:
        name_json = "vqa_" + name_json
    path_rslt = os.path.join(dir_epoch, name_json)
    os.system("mkdir -p " + dir_epoch)
    with open(path_rslt, "w") as handle:
        json.dump(results, handle)
    if not "test" in split_name:
        os.system(
            "python main/eval_res.py --dir_vqa {} --dir_epoch {} --subtype {} &".format(
                dir_vqa, dir_epoch, split_name
            )
        )


def save_checkpoint(
    info, model, optim, dir_logs, save_model, save_all_from=None, is_best=True
):
    os.system("mkdir -p " + dir_logs)
    if save_all_from is None:
        path_ckpt_info = os.path.join(dir_logs, "ckpt_info.pth.tar")
        path_ckpt_model = os.path.join(dir_logs, "ckpt_model.pth.tar")
        path_ckpt_optim = os.path.join(dir_logs, "ckpt_optim.pth.tar")
        path_best_info = os.path.join(dir_logs, "best_info.pth.tar")
        path_best_model = os.path.join(dir_logs, "best_model.pth.tar")
        path_best_optim = os.path.join(dir_logs, "best_optim.pth.tar")
        # save info & logger
        path_logger = os.path.join(dir_logs, "logger.json")
        info["exp_logger"].to_json(path_logger)
        torch.save(info, path_ckpt_info)
        if is_best:
            shutil.copyfile(path_ckpt_info, path_best_info)
        #  save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model)
            torch.save(optim, path_ckpt_optim)
            if is_best:
                shutil.copyfile(path_ckpt_model, path_best_model)
                shutil.copyfile(path_ckpt_optim, path_best_optim)
    else:
        is_best = False  # because we don't know the test accuracy
        path_ckpt_info = os.path.join(dir_logs, "ckpt_epoch,{}_info.pth.tar")
        path_ckpt_model = os.path.join(dir_logs, "ckpt_epoch,{}_model.pth.tar")
        path_ckpt_optim = os.path.join(dir_logs, "ckpt_epoch,{}_optim.pth.tar")
        # save info & logger
        path_logger = os.path.join(dir_logs, "logger.json")
        info["exp_logger"].to_json(path_logger)
        torch.save(info, path_ckpt_info.format(info["epoch"]))
        #  save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model.format(info["epoch"]))
            torch.save(optim, path_ckpt_optim.format(info["epoch"]))
        if info["epoch"] > 1 and info["epoch"] < save_all_from + 1:
            os.system("rm " + path_ckpt_info.format(info["epoch"] - 1))
            os.system("rm " + path_ckpt_model.format(info["epoch"] - 1))
            os.system("rm " + path_ckpt_optim.format(info["epoch"] - 1))
    if not save_model:
        print("Warning train.py: checkpoint not saved")


def load_checkpoint(model, optimizer, path_ckpt):
    is_loadable = True
    start_epoch = 0
    best_metric = 0
    exp_logger = None
    path_ckpt_info = path_ckpt + "_info.pth.tar"
    path_ckpt_model = path_ckpt + "_model.pth.tar"
    path_ckpt_optim = path_ckpt + "_optim.pth.tar"
    print("---------------------------------------------")
    print(path_ckpt_info)
    print(path_ckpt_model)
    print(path_ckpt_optim)
    print("---------------------------------------------")
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
        start_epoch = 0
        best_metric = 0
        exp_logger = None
        if "epoch" in info:
            start_epoch = info["epoch"]
        else:
            print("Warning train.py: no epoch to resume")
        if "best_acc1" in info:
            best_metric = info["best_acc1"]
        elif "best_metric" in info:
            best_metric = info["best_metric"]
        else:
            print("Warning train.py: no best_metric to resume")
        if "exp_logger" in info:
            exp_logger = info["exp_logger"]
        else:
            print("Warning train.py: no exp_logger to resume")
    else:
        print(
            "Warning train.py: no info checkpoint found at '{}'".format(path_ckpt_info)
        )
        is_loadable = False
    if os.path.isfile(path_ckpt_model):
        model_state = torch.load(path_ckpt_model)
        model.load_state_dict(model_state)
    else:
        print(
            "Warning train.py: no model checkpoint found at '{}'".format(
                path_ckpt_model
            )
        )
        is_loadable = False
    if optimizer is not None and os.path.isfile(path_ckpt_optim):
        optim_state = torch.load(path_ckpt_optim)
        optimizer.load_state_dict(optim_state)
    else:
        print(
            "Warning train.py: no optim checkpoint found at '{}'".format(
                path_ckpt_optim
            )
        )
        is_loadable = False
    print(
        "=> loaded checkpoint '{}' (epoch {}, best_metric {})".format(
            path_ckpt, start_epoch, best_metric
        )
    )
    start_epoch += 1
    return start_epoch, best_metric, exp_logger, is_loadable


def flatten(d, parent_key="", sep="_"):
    import collections

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,), mode="train"):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        if mode == "train":
            res = []
            for k in topk:
                # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        elif mode == "val":
            res = []
            k = 1
            correct_k = correct[:k].reshape(-1).float()
            res.extend(correct_k.tolist())
            return res


def make_meters(options):
    if options["data"]["dataset"] in ["mnist", "cifar10", "cifar100", "imagenet"]:
        meters_dict = {
            "loss": logger.AvgMeter(),
            "acc1": logger.AvgMeter(),
            "acc5": logger.AvgMeter(),
            "batch_time": logger.AvgMeter(),
            "data_time": logger.AvgMeter(),
            "epoch_time": logger.ValueMeter(),
            "max_memory_allocated": logger.ValueMeter(),
            "max_memory_cached": logger.ValueMeter(),
            "memory_allocated_before_forward": logger.AvgMeter(),
            "memory_cached_before_forward": logger.AvgMeter(),
            "memory_allocated_after_forward": logger.AvgMeter(),
            "memory_cached_after_forward": logger.AvgMeter(),
            "memory_allocated_after_backward": logger.AvgMeter(),
            "memory_cached_after_backward": logger.AvgMeter(),
        }
    elif options["data"]["dataset"] in ["brats", "spleen"]:
        meters_dict = {
            "loss": logger.AvgMeter(),
            "dice": logger.AvgMeter(),
            "batch_time": logger.AvgMeter(),
            "data_time": logger.AvgMeter(),
            "epoch_time": logger.ValueMeter(),
            "max_memory_allocated": logger.ValueMeter(),
            "max_memory_cached": logger.ValueMeter(),
            "memory_allocated_before_forward": logger.AvgMeter(),
            "memory_cached_before_forward": logger.AvgMeter(),
            "memory_allocated_after_forward": logger.AvgMeter(),
            "memory_cached_after_forward": logger.AvgMeter(),
            "memory_allocated_after_backward": logger.AvgMeter(),
            "memory_cached_after_backward": logger.AvgMeter(),
        }

    return meters_dict


class Classification:
    def train(
        self,
        train_loader,
        model,
        criterion,
        optimizer,
        exp_logger,
        epoch,
        experiment,
        args,
        options,
        print_freq=10,
    ):
        # switch to train mode
        model.train()
        meters = exp_logger.reset_meters("train")
        end = time.time()
        start_epoch = time.time()
        batch_size = options["optim"]["batch_size"]

        start_time = time.time()

        # if "compressed" in options["model"]["arch"]:
        #     wave = options["model"]["wave"]
        #     compression_ratio = options["model"]["compression_ratio"]
        #     n_levels = options["model"]["n_levels"]

        for i, (images, target) in enumerate(train_loader):
            # memory_utils.print_memory(verbose=1)

            # measure data loading time
            meters["data_time"].update(time.time() - end, n=batch_size)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # get memory before forward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()
            _, _, memory_allocated, memory_cached = memory_utils.print_memory()
            meters["memory_allocated_before_forward"].update(memory_allocated)
            meters["memory_cached_before_forward"].update(memory_cached)

            # compute output
            # if "compressed" in options["model"]["arch"]:
            #     output = model(E_epoch, T_epoch, wave, compression_ratio, n_levels, images)
            # else:
            output = model(images)

            loss = criterion(output, target)

            # get memory after forward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()
            _, _, memory_allocated, memory_cached = memory_utils.print_memory()
            meters["memory_allocated_after_forward"].update(memory_allocated)
            meters["memory_cached_after_forward"].update(memory_cached)

            # memory_utils.print_memory(verbose=1)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            meters["loss"].update(loss.item(), images.size(0))
            meters["acc1"].update(acc1[0].item(), images.size(0))
            meters["acc5"].update(acc5[0].item(), images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()

            # params_before = []
            # for p in list(model.parameters()):
            #     params_before.append(p.clone())

            loss.backward()
            # memory_utils.getBack(loss.grad_fn)

            optimizer.step()

            # params_after = []
            # for p in list(model.parameters()):
            #     params_after.append(p.clone())

            # for i in range(8):
            #     # print(a[i].data, b[i].data)
            #     print(torch.equal(params_before[i].data, params_after[i].data))

            # get memory after backward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()

            (
                max_memory_allocated,
                max_memory_cached,
                memory_allocated,
                memory_cached,
            ) = memory_utils.print_memory()
            meters["memory_allocated_after_backward"].update(memory_allocated)
            meters["memory_cached_after_backward"].update(memory_cached)
            meters["max_memory_allocated"].update(max_memory_allocated)
            meters["max_memory_cached"].update(max_memory_cached)

            # measure elapsed time
            meters["batch_time"].update(time.time() - end)
            end = time.time()

            # memory_utils.print_memory(verbose=1)

            if i % print_freq == 0:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t"
                    "Acc@5 {acc5.val:.3f} ({acc5.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        loss=meters["loss"],
                        acc1=meters["acc1"],
                        acc5=meters["acc5"],
                    )
                )

                # Log to Comet.ml
                if experiment is not None:
                    experiment.log_metric("batch_acc1", meters["acc1"].avg)

        # Log to Comet.ml
        if experiment is not None:
            experiment.log_metric("acc1", meters["acc1"].avg)

        meters["epoch_time"].update(time.time() - start_epoch)
        exp_logger.log_meters("train", n=epoch)

        end_time = time.time() - start_time
        print("Epoch: [{0}]\t" "Epoch time {1}s".format(epoch, end_time))

    def validiate(
        self,
        val_loader,
        model,
        criterion,
        exp_logger,
        args,
        options,
        epoch=0,
        print_freq=10,
    ):
        # switch to evaluate mode
        model.eval()
        meters = exp_logger.reset_meters("test")

        end = time.time()
        start_epoch = time.time()

        # if "compressed" in options["model"]["arch"]:
        #     wave = options["model"]["wave"]
        #     compression_ratio = options["model"]["compression_ratio"]
        #     n_levels = options["model"]["n_levels"]

        with torch.no_grad():
            end = time.time()
            batch_size = options["optim"]["batch_size"]
            for i, (images, target) in enumerate(val_loader):
                meters["data_time"].update(time.time() - end, n=batch_size)
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                # get memory before forward
                _, _, memory_allocated, memory_cached = memory_utils.print_memory()
                meters["memory_allocated_before_forward"].update(memory_allocated)
                meters["memory_cached_before_forward"].update(memory_cached)

                # compute output
                # if "compressed" in options["model"]["arch"]:
                #     _, _, output = model(
                #         None, None, wave, compression_ratio, n_levels, images
                #     )
                # else:
                output = model(images)
                loss = criterion(output, target)

                # get memory after forward
                (
                    max_memory_allocated,
                    max_memory_cached,
                    memory_allocated,
                    memory_cached,
                ) = memory_utils.print_memory()
                meters["memory_allocated_after_forward"].update(memory_allocated)
                meters["memory_cached_after_forward"].update(memory_cached)
                meters["max_memory_allocated"].update(max_memory_allocated)
                meters["max_memory_cached"].update(max_memory_cached)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                meters["loss"].update(loss.item(), images.size(0))
                meters["acc1"].update(acc1[0].item(), images.size(0))
                meters["acc5"].update(acc5[0].item(), images.size(0))

                # measure elapsed time
                meters["batch_time"].update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    print(
                        "Val: [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        #   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        "Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t"
                        "Acc@5 {acc5.val:.3f} ({acc5.avg:.3f})".format(
                            i,
                            len(val_loader),
                            batch_time=meters["batch_time"],
                            data_time=meters["data_time"],
                            acc1=meters["acc1"],
                            acc5=meters["acc5"],
                        )
                    )

        print(
            " * Acc@1 {acc1.avg:.3f} Acc@5 {acc5.avg:.3f}".format(
                acc1=meters["acc1"], acc5=meters["acc5"]
            )
        )

        meters["epoch_time"].update(time.time() - start_epoch)
        exp_logger.log_meters("test", n=epoch)
        return meters["acc1"].avg

    def train_all_epoch(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        for epoch in range(args.start_epoch, options["optim"]["epochs"] + 1):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, options["optim"]["learning_rate"])
            self.train(
                train_loader,
                model,
                criterion,
                optimizer,
                exp_logger,
                epoch,
                experiment,
                args,
                options,
                print_freq=args.print_freq,
            )

            if experiment is not None:
                with experiment.validate():
                    acc1 = self.validiate(
                        val_loader,
                        model,
                        criterion,
                        exp_logger,
                        args,
                        options,
                        epoch=epoch,
                        print_freq=args.print_freq,
                    )
                    experiment.log_metric("acc1", acc1)
            else:
                acc1 = self.validiate(
                    val_loader,
                    model,
                    criterion,
                    exp_logger,
                    args,
                    options,
                    epoch=epoch,
                    print_freq=args.print_freq,
                )

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_metric
            best_metric = max(acc1, best_metric)

            if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
            ):
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "arch": options["model"]["arch"],
                        "best_metric": best_metric,
                        "exp_logger": exp_logger,
                    },
                    model.state_dict(),
                    optimizer.state_dict(),
                    options["logs"]["dir_logs"],
                    args.save_model,
                    args.save_all_from,
                    is_best,
                )

        return best_metric

    def perform_train_validiate(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        # if experiment is not None:
        best_metric = self.train_all_epoch(
            args,
            options,
            train_sampler,
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            exp_logger,
            experiment,
            ngpus_per_node,
            best_metric,
        )

        return best_metric

    def __validiate__(
        self,
        val_loader,
        model,
        criterion,
        exp_logger,
        args,
        options,
        epoch=0,
        print_freq=10,
    ):
        # switch to evaluate mode
        model.eval()
        list_acc1 = []

        with torch.no_grad():
            end = time.time()
            batch_size = options["optim"]["batch_size"]
            for i, (images, target) in enumerate(val_loader):
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1 = accuracy(output, target, topk=(1, 5), mode="val")
                list_acc1.extend(acc1)

        return list_acc1

    def validiate_eval_mode(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        for epoch in range(1):
            list_acc1 = self.__validiate__(
                val_loader,
                model,
                criterion,
                exp_logger,
                args,
                options,
                epoch=epoch,
                print_freq=args.print_freq,
            )

        return list_acc1

    def perform_validiate(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        # if experiment is not None:
        best_metric = self.validiate_eval_mode(
            args,
            options,
            train_sampler,
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            exp_logger,
            experiment,
            ngpus_per_node,
            best_metric,
        )

        return best_metric


class SegmentationBratsNew:
    def train(
        self,
        train_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        exp_logger,
        epoch,
        experiment,
        args,
        options,
        print_freq=10,
    ):
        # switch to train mode
        model.train()
        meters = exp_logger.reset_meters("train")
        end = time.time()
        start_epoch = time.time()
        batch_size = options["optim"]["batch_size"]
        start_time = time.time()

        epoch_loss = 0
        step = 0

        for i, batch_data in enumerate(train_loader):
            step += 1
            inputs, labels = (
                batch_data["image"].cuda(args.gpu, non_blocking=True),
                batch_data["label"].cuda(args.gpu, non_blocking=True),
            )
            # measure data loading time
            meters["data_time"].update(time.time() - end, n=batch_size)

            # get memory before forward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()
            _, _, memory_allocated, memory_cached = memory_utils.print_memory()
            meters["memory_allocated_before_forward"].update(memory_allocated)
            meters["memory_cached_before_forward"].update(memory_cached)

            optimizer.zero_grad()
            # outputs = model(inputs).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # get memory after forward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()
            _, _, memory_allocated, memory_cached = memory_utils.print_memory()
            meters["memory_allocated_after_forward"].update(memory_allocated)
            meters["memory_cached_after_forward"].update(memory_cached)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            meters["loss"].update(loss.item(), inputs.size(0))
            meters["dice"].update(1.0 - loss.item(), inputs.size(0))

            # get memory after backward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()

            (
                max_memory_allocated,
                max_memory_cached,
                memory_allocated,
                memory_cached,
            ) = memory_utils.print_memory()
            meters["memory_allocated_after_backward"].update(memory_allocated)
            meters["memory_cached_after_backward"].update(memory_cached)
            meters["max_memory_allocated"].update(max_memory_allocated)
            meters["max_memory_cached"].update(max_memory_cached)

            # measure elapsed time
            meters["batch_time"].update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Dice {dice.val:.3f} ({dice.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        loss=meters["loss"],
                        dice=meters["dice"],
                    )
                )

                # Log to Comet.ml
                if experiment is not None:
                    experiment.log_metric("batch_dice", meters["dice"].avg)

        lr_scheduler.step()
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Log to Comet.ml
        if experiment is not None:
            experiment.log_metric("dice", meters["dice"].avg)

        meters["epoch_time"].update(time.time() - start_epoch)
        exp_logger.log_meters("train", n=epoch)

        end_time = time.time() - start_time
        print("Epoch: [{0}]\t" "Epoch time {1}s".format(epoch, end_time))

    # define inference method
    def inference(self, model, input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        with torch.cuda.amp.autocast():
            return _compute(input)

    def validiate(
        self,
        val_loader,
        model,
        criterion,
        exp_logger,
        args,
        options,
        epoch=0,
        print_freq=10,
    ):
        # switch to evaluate mode
        model.eval()
        meters = exp_logger.reset_meters("test")

        end = time.time()
        start_epoch = time.time()

        post_trans = Compose(
            [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
        )

        with torch.no_grad():
            end = time.time()
            batch_size = options["optim"]["batch_size"]
            for i, batch_data in enumerate(val_loader):
                step_start = time.time()
                inputs, labels = (
                    batch_data["image"].cuda(args.gpu, non_blocking=True),
                    batch_data["label"].cuda(args.gpu, non_blocking=True),
                )
                # measure data loading time
                meters["data_time"].update(time.time() - end, n=batch_size)

                # get memory before forward
                if args.empty_memory_cache:
                    torch.cuda.empty_cache()
                _, _, memory_allocated, memory_cached = memory_utils.print_memory()
                meters["memory_allocated_before_forward"].update(memory_allocated)
                meters["memory_cached_before_forward"].update(memory_cached)

                # outputs = model(inputs)
                outputs = self.inference(model, inputs)
                outputs = [post_trans(i) for i in decollate_batch(outputs)]

                dice_metric = DiceMetric(include_background=True, reduction="mean")
                dice = dice_metric(y_pred=outputs, y=labels)

                # get memory after forward
                if args.empty_memory_cache:
                    torch.cuda.empty_cache()
                _, _, memory_allocated, memory_cached = memory_utils.print_memory()
                meters["memory_allocated_after_forward"].update(memory_allocated)
                meters["memory_cached_after_forward"].update(memory_cached)

                meters["loss"].update(1 - dice[0].mean().item(), inputs.size(0))
                meters["dice"].update(dice[0].mean().item(), inputs.size(0))

                # get memory after backward
                if args.empty_memory_cache:
                    torch.cuda.empty_cache()

                (
                    max_memory_allocated,
                    max_memory_cached,
                    memory_allocated,
                    memory_cached,
                ) = memory_utils.print_memory()
                meters["memory_allocated_after_backward"].update(memory_allocated)
                meters["memory_cached_after_backward"].update(memory_cached)
                meters["max_memory_allocated"].update(max_memory_allocated)
                meters["max_memory_cached"].update(max_memory_cached)

                # measure elapsed time
                meters["batch_time"].update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Dice {dice.val:.3f} ({dice.avg:.3f})\t".format(
                            epoch,
                            i,
                            len(val_loader),
                            loss=meters["loss"],
                            dice=meters["dice"],
                        )
                    )

            print(" * Dice {dice.avg:.3f}".format(dice=meters["dice"]))

            meters["epoch_time"].update(time.time() - start_epoch)
            exp_logger.log_meters("test", n=epoch)
            return meters["dice"].avg

    def train_all_epoch(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        for epoch in range(args.start_epoch, options["optim"]["epochs"] + 1):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, options["optim"]["learning_rate"])
            self.train(
                train_loader,
                model,
                criterion,
                optimizer,
                lr_scheduler,
                exp_logger,
                epoch,
                experiment,
                args,
                options,
                print_freq=10,
            )

            if experiment is not None:
                with experiment.validate():
                    metric = self.validiate(
                        val_loader,
                        model,
                        criterion,
                        exp_logger,
                        args,
                        options,
                        epoch=epoch,
                        print_freq=10,
                    )
                    experiment.log_metric("dice", metric)
            else:
                metric = self.validiate(
                    val_loader,
                    model,
                    criterion,
                    exp_logger,
                    args,
                    options,
                    epoch=epoch,
                    print_freq=10,
                )

            # remember best acc@1 and save checkpoint
            is_best = metric > best_metric
            best_metric = max(metric, best_metric)

            if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
            ):
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "arch": options["model"]["arch"],
                        "best_metric": best_metric,
                        "exp_logger": exp_logger,
                    },
                    model.state_dict(),
                    optimizer.state_dict(),
                    options["logs"]["dir_logs"],
                    args.save_model,
                    args.save_all_from,
                    is_best,
                )

        return best_metric

    def perform_train_validiate(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        # if experiment is not None:
        best_metric = self.train_all_epoch(
            args,
            options,
            train_sampler,
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            exp_logger,
            experiment,
            ngpus_per_node,
            best_metric,
        )

        return best_metric


class SegmentationBrats:
    def train(
        self,
        train_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        exp_logger,
        epoch,
        experiment,
        args,
        options,
        print_freq=10,
    ):
        scaler = torch.cuda.amp.GradScaler()
        # switch to train mode
        model.train()
        meters = exp_logger.reset_meters("train")
        end = time.time()
        start_epoch = time.time()
        batch_size = options["optim"]["batch_size"]
        start_time = time.time()

        epoch_loss = 0
        step = 0

        for i, batch_data in enumerate(train_loader):
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].cuda(args.gpu, non_blocking=True),
                batch_data["label"].cuda(args.gpu, non_blocking=True),
            )
            # measure data loading time
            meters["data_time"].update(time.time() - end, n=batch_size)

            # get memory before forward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()
            _, _, memory_allocated, memory_cached = memory_utils.print_memory()
            meters["memory_allocated_before_forward"].update(memory_allocated)
            meters["memory_cached_before_forward"].update(memory_cached)

            optimizer.zero_grad()
            # with torch.cuda.amp.autocast(dtype=torch.float32):
            with torch.cuda.amp.autocast():
                # with torch.cuda.amp.autocast(dtype=torch.float16):
                # outputs = model(inputs).float()
                outputs = model(inputs)
                # outputs = model(inputs.float())
                loss = criterion(outputs, labels)

            # get memory after forward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()
            _, _, memory_allocated, memory_cached = memory_utils.print_memory()
            meters["memory_allocated_after_forward"].update(memory_allocated)
            meters["memory_cached_after_forward"].update(memory_cached)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            meters["loss"].update(loss.item(), inputs.size(0))
            meters["dice"].update(1.0 - loss.item(), inputs.size(0))

            # get memory after backward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()

            (
                max_memory_allocated,
                max_memory_cached,
                memory_allocated,
                memory_cached,
            ) = memory_utils.print_memory()
            meters["memory_allocated_after_backward"].update(memory_allocated)
            meters["memory_cached_after_backward"].update(memory_cached)
            meters["max_memory_allocated"].update(max_memory_allocated)
            meters["max_memory_cached"].update(max_memory_cached)

            # measure elapsed time
            meters["batch_time"].update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Dice {dice.val:.3f} ({dice.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        loss=meters["loss"],
                        dice=meters["dice"],
                    )
                )

                # Log to Comet.ml
                if experiment is not None:
                    experiment.log_metric("batch_dice", meters["dice"].avg)

        lr_scheduler.step()
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Log to Comet.ml
        if experiment is not None:
            experiment.log_metric("dice", meters["dice"].avg)

        meters["epoch_time"].update(time.time() - start_epoch)
        exp_logger.log_meters("train", n=epoch)

        end_time = time.time() - start_time
        print("Epoch: [{0}]\t" "Epoch time {1}s".format(epoch, end_time))

    # define inference method
    def inference(self, model, input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        with torch.cuda.amp.autocast():
            return _compute(input)

    def validiate(
        self,
        val_loader,
        model,
        criterion,
        exp_logger,
        args,
        options,
        epoch=0,
        print_freq=10,
    ):
        # switch to evaluate mode
        model.eval()
        meters = exp_logger.reset_meters("test")

        end = time.time()
        start_epoch = time.time()

        post_trans = Compose(
            [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
        )

        with torch.no_grad():
            end = time.time()
            batch_size = options["optim"]["batch_size"]
            for i, batch_data in enumerate(val_loader):
                step_start = time.time()
                inputs, labels = (
                    batch_data["image"].cuda(args.gpu, non_blocking=True),
                    batch_data["label"].cuda(args.gpu, non_blocking=True),
                )
                # measure data loading time
                meters["data_time"].update(time.time() - end, n=batch_size)

                # get memory before forward
                if args.empty_memory_cache:
                    torch.cuda.empty_cache()
                _, _, memory_allocated, memory_cached = memory_utils.print_memory()
                meters["memory_allocated_before_forward"].update(memory_allocated)
                meters["memory_cached_before_forward"].update(memory_cached)

                # outputs = model(inputs)
                outputs = self.inference(model, inputs)
                outputs = [post_trans(i) for i in decollate_batch(outputs)]

                dice_metric = DiceMetric(include_background=True, reduction="mean")
                dice = dice_metric(y_pred=outputs, y=labels)

                # get memory after forward
                if args.empty_memory_cache:
                    torch.cuda.empty_cache()
                _, _, memory_allocated, memory_cached = memory_utils.print_memory()
                meters["memory_allocated_after_forward"].update(memory_allocated)
                meters["memory_cached_after_forward"].update(memory_cached)

                meters["loss"].update(1 - dice[0].mean().item(), inputs.size(0))
                meters["dice"].update(dice[0].mean().item(), inputs.size(0))

                # get memory after backward
                if args.empty_memory_cache:
                    torch.cuda.empty_cache()

                (
                    max_memory_allocated,
                    max_memory_cached,
                    memory_allocated,
                    memory_cached,
                ) = memory_utils.print_memory()
                meters["memory_allocated_after_backward"].update(memory_allocated)
                meters["memory_cached_after_backward"].update(memory_cached)
                meters["max_memory_allocated"].update(max_memory_allocated)
                meters["max_memory_cached"].update(max_memory_cached)

                # measure elapsed time
                meters["batch_time"].update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Dice {dice.val:.3f} ({dice.avg:.3f})\t".format(
                            epoch,
                            i,
                            len(val_loader),
                            loss=meters["loss"],
                            dice=meters["dice"],
                        )
                    )

            print(" * Dice {dice.avg:.3f}".format(dice=meters["dice"]))

            meters["epoch_time"].update(time.time() - start_epoch)
            exp_logger.log_meters("test", n=epoch)
            return meters["dice"].avg

    def train_all_epoch(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        for epoch in range(args.start_epoch, options["optim"]["epochs"] + 1):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, options["optim"]["learning_rate"])
            self.train(
                train_loader,
                model,
                criterion,
                optimizer,
                lr_scheduler,
                exp_logger,
                epoch,
                experiment,
                args,
                options,
                print_freq=10,
            )

            if experiment is not None:
                with experiment.validate():
                    metric = self.validiate(
                        val_loader,
                        model,
                        criterion,
                        exp_logger,
                        args,
                        options,
                        epoch=epoch,
                        print_freq=10,
                    )
                    experiment.log_metric("dice", metric)
            else:
                metric = self.validiate(
                    val_loader,
                    model,
                    criterion,
                    exp_logger,
                    args,
                    options,
                    epoch=epoch,
                    print_freq=10,
                )

            # remember best acc@1 and save checkpoint
            is_best = metric > best_metric
            best_metric = max(metric, best_metric)

            if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
            ):
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "arch": options["model"]["arch"],
                        "best_metric": best_metric,
                        "exp_logger": exp_logger,
                    },
                    model.state_dict(),
                    optimizer.state_dict(),
                    options["logs"]["dir_logs"],
                    args.save_model,
                    args.save_all_from,
                    is_best,
                )

        return best_metric

    def perform_train_validiate(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        # if experiment is not None:
        best_metric = self.train_all_epoch(
            args,
            options,
            train_sampler,
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            exp_logger,
            experiment,
            ngpus_per_node,
            best_metric,
        )

        return best_metric

    def __validiate__(
        self,
        val_loader,
        model,
        criterion,
        exp_logger,
        args,
        options,
        epoch=0,
        print_freq=10,
    ):
        # switch to evaluate mode
        model.eval()

        post_trans = Compose(
            [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
        )

        list_dice = []

        with torch.no_grad():
            batch_size = options["optim"]["batch_size"]
            for i, batch_data in enumerate(val_loader):
                print(f"{i} / {len(val_loader)}")
                inputs, labels = (
                    batch_data["image"].cuda(args.gpu, non_blocking=True),
                    batch_data["label"].cuda(args.gpu, non_blocking=True),
                )

                # outputs = model(inputs)
                outputs = self.inference(model, inputs)
                outputs = [post_trans(i) for i in decollate_batch(outputs)]

                dice_metric = DiceMetric(include_background=True, reduction="mean")
                dice = dice_metric(y_pred=outputs, y=labels)

                for i in range(batch_size):
                    list_dice.append(dice[i].mean().item())

            return list_dice

    def validiate_eval_mode(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        for epoch in range(1):
            list_metric = self.__validiate__(
                val_loader,
                model,
                criterion,
                exp_logger,
                args,
                options,
                epoch=epoch,
                print_freq=10,
            )

        return list_metric

    def perform_validiate(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        # if experiment is not None:
        best_metric = self.validiate_eval_mode(
            args,
            options,
            train_sampler,
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            exp_logger,
            experiment,
            ngpus_per_node,
            best_metric,
        )

        return best_metric


class Segmentation:
    def train(
        self,
        train_loader,
        model,
        criterion,
        optimizer,
        exp_logger,
        epoch,
        experiment,
        args,
        options,
        print_freq=1,
    ):
        model.train()
        meters = exp_logger.reset_meters("train")
        end = time.time()
        start_epoch = time.time()
        batch_size = options["optim"]["batch_size"]
        start_time = time.time()

        epoch_loss = 0
        step = 0

        print("-" * 20)
        for i, batch_data in enumerate(train_loader):
            step += 1
            inputs, labels = (
                batch_data["image"].cuda(args.gpu, non_blocking=True),
                batch_data["label"].cuda(args.gpu, non_blocking=True),
            )
            # measure data loading time
            meters["data_time"].update(time.time() - end, n=batch_size)

            # get memory before forward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()
            _, _, memory_allocated, memory_cached = memory_utils.print_memory()
            meters["memory_allocated_before_forward"].update(memory_allocated)
            meters["memory_cached_before_forward"].update(memory_cached)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # get memory after forward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()
            _, _, memory_allocated, memory_cached = memory_utils.print_memory()
            meters["memory_allocated_after_forward"].update(memory_allocated)
            meters["memory_cached_after_forward"].update(memory_cached)

            meters["loss"].update(loss.item(), inputs.size(0))
            meters["dice"].update(1.0 - loss.item(), inputs.size(0))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # get memory after backward
            if args.empty_memory_cache:
                torch.cuda.empty_cache()

            (
                max_memory_allocated,
                max_memory_cached,
                memory_allocated,
                memory_cached,
            ) = memory_utils.print_memory()
            meters["memory_allocated_after_backward"].update(memory_allocated)
            meters["memory_cached_after_backward"].update(memory_cached)
            meters["max_memory_allocated"].update(max_memory_allocated)
            meters["max_memory_cached"].update(max_memory_cached)

            # measure elapsed time
            meters["batch_time"].update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Dice {dice.val:.3f} ({dice.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        loss=meters["loss"],
                        dice=meters["dice"],
                    )
                )

                # Log to Comet.ml
                if experiment is not None:
                    experiment.log_metric("batch_dice", meters["dice"].avg)

        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Log to Comet.ml
        if experiment is not None:
            experiment.log_metric("dice", meters["dice"].avg)

        meters["epoch_time"].update(time.time() - start_epoch)
        exp_logger.log_meters("train", n=epoch)

        end_time = time.time() - start_time
        print("Epoch: [{0}]\t" "Epoch time {1}s".format(epoch, end_time))

    def validiate(
        self,
        val_loader,
        model,
        exp_logger,
        args,
        options,
        epoch=0,
        print_freq=1,
    ):
        # switch to evaluate mode
        model.eval()
        meters = exp_logger.reset_meters("test")

        end = time.time()
        start_epoch = time.time()

        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        roi_size = (160, 160, 160)

        with torch.no_grad():
            end = time.time()
            batch_size = options["optim"]["batch_size"]
            for i, batch_data in enumerate(val_loader):
                inputs, labels = (
                    batch_data["image"].cuda(args.gpu, non_blocking=True),
                    batch_data["label"].cuda(args.gpu, non_blocking=True),
                )
                # measure data loading time
                meters["data_time"].update(time.time() - end, n=batch_size)

                # get memory before forward
                if args.empty_memory_cache:
                    torch.cuda.empty_cache()
                _, _, memory_allocated, memory_cached = memory_utils.print_memory()
                meters["memory_allocated_before_forward"].update(memory_allocated)
                meters["memory_cached_before_forward"].update(memory_cached)

                sw_batch_size = 4
                outputs = sliding_window_inference(
                    inputs, roi_size, sw_batch_size, model
                )
                outputs = [post_pred(i) for i in decollate_batch(outputs)]
                labels = [post_label(i) for i in decollate_batch(labels)]

                dice_metric = DiceMetric(include_background=True, reduction="mean")
                dice = dice_metric(y_pred=outputs, y=labels)

                # get memory after forward
                if args.empty_memory_cache:
                    torch.cuda.empty_cache()
                _, _, memory_allocated, memory_cached = memory_utils.print_memory()
                meters["memory_allocated_after_forward"].update(memory_allocated)
                meters["memory_cached_after_forward"].update(memory_cached)

                meters["loss"].update(1 - dice[0].mean().item(), inputs.size(0))
                meters["dice"].update(dice[0].mean().item(), inputs.size(0))

                # get memory after backward
                if args.empty_memory_cache:
                    torch.cuda.empty_cache()

                (
                    max_memory_allocated,
                    max_memory_cached,
                    memory_allocated,
                    memory_cached,
                ) = memory_utils.print_memory()
                meters["memory_allocated_after_backward"].update(memory_allocated)
                meters["memory_cached_after_backward"].update(memory_cached)
                meters["max_memory_allocated"].update(max_memory_allocated)
                meters["max_memory_cached"].update(max_memory_cached)

                # measure elapsed time
                meters["batch_time"].update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Dice {dice.val:.3f} ({dice.avg:.3f})\t".format(
                            epoch,
                            i,
                            len(val_loader),
                            loss=meters["loss"],
                            dice=meters["dice"],
                        )
                    )

            print(" * Dice {dice.avg:.3f}".format(dice=meters["dice"]))

            meters["epoch_time"].update(time.time() - start_epoch)
            exp_logger.log_meters("test", n=epoch)
            return meters["dice"].avg

    def train_all_epoch(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        for epoch in range(args.start_epoch, options["optim"]["epochs"] + 1):
            self.train(
                train_loader,
                model,
                criterion,
                optimizer,
                exp_logger,
                epoch,
                experiment,
                args,
                options,
                print_freq=1,
            )

            if experiment is not None:
                with experiment.validate():
                    metric = self.validiate(
                        val_loader,
                        model,
                        exp_logger,
                        args,
                        options,
                        epoch=epoch,
                        print_freq=1,
                    )
                    experiment.log_metric("dice", metric)
            else:
                metric = self.validiate(
                    val_loader,
                    model,
                    exp_logger,
                    args,
                    options,
                    epoch=epoch,
                    print_freq=1,
                )

            # remember best acc@1 and save checkpoint
            is_best = metric > best_metric
            best_metric = max(metric, best_metric)

            if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
            ):
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "arch": options["model"]["arch"],
                        "best_metric": best_metric,
                        "exp_logger": exp_logger,
                    },
                    model.state_dict(),
                    optimizer.state_dict(),
                    options["logs"]["dir_logs"],
                    args.save_model,
                    args.save_all_from,
                    is_best,
                )

        return best_metric

    def perform_train_validiate(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        # if experiment is not None:
        best_metric = self.train_all_epoch(
            args,
            options,
            train_sampler,
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            exp_logger,
            experiment,
            ngpus_per_node,
            best_metric,
        )

        return best_metric

    def __validiate__(
        self,
        val_loader,
        model,
        exp_logger,
        args,
        options,
        epoch=0,
        print_freq=1,
    ):
        # switch to evaluate mode
        model.eval()

        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        roi_size = (160, 160, 160)

        list_dice = []

        with torch.no_grad():
            batch_size = options["optim"]["batch_size"]
            for i, batch_data in enumerate(val_loader):
                inputs, labels = (
                    batch_data["image"].cuda(args.gpu, non_blocking=True),
                    batch_data["label"].cuda(args.gpu, non_blocking=True),
                )

                sw_batch_size = 4
                outputs = sliding_window_inference(
                    inputs, roi_size, sw_batch_size, model
                )
                outputs = [post_pred(i) for i in decollate_batch(outputs)]
                labels = [post_label(i) for i in decollate_batch(labels)]

                dice_metric = DiceMetric(include_background=True, reduction="mean")
                dice = dice_metric(y_pred=outputs, y=labels)

                for i in range(batch_size):
                    list_dice.append(dice[i].mean().item())

            return list_dice

    def validiate_eval_mode(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        for epoch in range(1):
            metric = self.__validiate__(
                val_loader,
                model,
                exp_logger,
                args,
                options,
                epoch=epoch,
                print_freq=1,
            )

        return metric

    def perform_validiate(
        self,
        args,
        options,
        train_sampler,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        exp_logger,
        experiment,
        ngpus_per_node,
        best_metric,
    ):
        # if experiment is not None:
        best_metric = self.validiate_eval_mode(
            args,
            options,
            train_sampler,
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            exp_logger,
            experiment,
            ngpus_per_node,
            best_metric,
        )

        return best_metric
