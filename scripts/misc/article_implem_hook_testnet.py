import pandas as pd
import torch
from torch import nn

from engine.utils.memory_utils import (
    log_mem,
)
from engine.utils.plot_utils import plot_mem

from models.clasification.testnet import (
    MNISTNet,
    MNISTNetCompressed,
    MNISTNet_Compressed_Hook_Silly,
)


# Create Sequential version of model
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


def test():
    base_dir = "figures"

    bs = 128
    input = torch.rand(bs, 32, 128, 128, requires_grad=True).cuda()
    mem_log = []

    # Baseline
    model = MNISTNet().cuda()
    try:
        mem_log.extend(log_mem(model, input, exp="baseline"))
    except Exception as e:
        print(f"log_mem failed because of {e}")

    df = pd.DataFrame(mem_log)

    # pp(df, exp="baseline")

    # Hook
    model_ckpt = MNISTNet_Compressed_Hook_Silly().cuda()
    try:
        mem_log.extend(log_mem(model_ckpt, input, exp="silly-hook"))
    except Exception as e:
        print(f"log_mem_cp failed because of {e}")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    df = pd.DataFrame(mem_log)

    # Hook
    model_ckpt = MNISTNetCompressed().cuda()
    try:
        mem_log.extend(log_mem(model_ckpt, input, exp="hook"))
    except Exception as e:
        print(f"log_mem_cp failed because of {e}")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Plot all files
    df = pd.DataFrame(mem_log)
    plot_mem(df, output_file=f"{base_dir}/mnistnet_{bs}.png")


if __name__ == "__main__":
    # example()
    test()
