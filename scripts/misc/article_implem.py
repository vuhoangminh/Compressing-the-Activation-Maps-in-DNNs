import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18

from engine.utils.memory_utils import (
    log_mem,
    log_mem_amp,
    log_mem_amp_cp,
    log_mem_cp,
    log_mem_compressed,
)
from engine.utils.plot_utils import plot_mem, pp

from models.clasification.mnistnet import (
    MNISTNet,
    MNISTNet_ReLU,
    MNISTNet_ReLU_Custom,
    MNISTNet_ReLU_Compressed,
)


# Create Sequential version of model
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


def example():
    base_dir = "figures"
    # Analysis baseline

    model = resnet18().cuda()
    bs = 128
    input = torch.rand(bs, 3, 224, 224).cuda()

    mem_log = []

    try:
        mem_log.extend(log_mem(model, input, exp="baseline"))
    except Exception as e:
        print(f"log_mem failed because of {e}")

    df = pd.DataFrame(mem_log)

    plot_mem(
        df, exps=["baseline"], output_file=f"{base_dir}/baseline_memory_plot_{bs}.png"
    )

    pp(df, exp="baseline")

    seq_model = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        Flatten(),
        model.fc,
    )

    # Test models are identical:

    with torch.no_grad():
        out = model(input)
        seq_out = seq_model(input)
        max_diff = (out - seq_out).max().abs().item()
        assert max_diff < 10**-10

    # Log mem optims

    try:
        mem_log.extend(log_mem_cp(seq_model, input, cp_chunks=3, exp="3_checkpoints"))
    except Exception as e:
        print(f"log_mem_cp failed because of {e}")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    try:
        mem_log.extend(log_mem_amp(model, input, exp="auto_mixed_precision"))
    except Exception as e:
        print(f"log_mem_amp failed because of {e}")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    try:
        mem_log.extend(
            log_mem_amp_cp(seq_model, input, cp_chunks=3, exp="amp_and_3_cp")
        )
    except Exception as e:
        print(f"log_mem_amp_cp failed because of {e}")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Plot all files

    df = pd.DataFrame(mem_log)

    plot_mem(df, output_file=f"{base_dir}/resnet50_all_memory_plot_{bs}.png")

    # Get max memory

    (
        df.groupby("exp")
        .mem_all.max()
        .sort_values(ascending=False)
        .plot("bar")
        .get_figure()
        .savefig(f"{base_dir}/resnet50_max_mem_hist_{bs}.png")
    )


def mnistnet():
    base_dir = "figures"

    # Baseline
    model = MNISTNet().cuda()
    bs = 128
    input = torch.rand(bs, 1, 28, 28).cuda()

    mem_log = []

    try:
        mem_log.extend(log_mem(model, input, exp="baseline"))
    except Exception as e:
        print(f"log_mem failed because of {e}")

    df = pd.DataFrame(mem_log)

    # plot_mem(
    #     df, exps=["baseline"], output_file=f"{base_dir}/baseline_memory_plot_{bs}.png"
    # )

    # pp(df, exp="baseline")

    # ReLU
    model_ckpt = MNISTNet_ReLU().cuda()
    try:
        mem_log.extend(log_mem(model_ckpt, input, exp="relu"))
    except Exception as e:
        print(f"log_mem_cp failed because of {e}")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ReLU Custom
    model_ckpt = MNISTNet_ReLU_Custom().cuda()
    try:
        mem_log.extend(log_mem(model_ckpt, input, exp="relu_custom"))
    except Exception as e:
        print(f"log_mem_cp failed because of {e}")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ReLU Compressed
    model_ckpt = MNISTNet_ReLU_Compressed().cuda()
    mem_log.extend(log_mem_compressed(model_ckpt, input, exp="relu_compressed"))

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Plot all files
    df = pd.DataFrame(mem_log)
    plot_mem(df, output_file=f"{base_dir}/mnistnet_{bs}.png")

    # # Get max memory

    # (
    #     df.groupby("exp")
    #     .mem_all.max()
    #     .sort_values(ascending=False)
    #     .plot("bar")
    #     .get_figure()
    #     .savefig(f"{base_dir}/resnet50_max_mem_hist_{bs}.png")
    # )


if __name__ == "__main__":
    # example()
    mnistnet()
