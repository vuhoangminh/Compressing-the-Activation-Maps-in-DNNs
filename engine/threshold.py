import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import engine.utils.print_utils as print_utils
import engine.utils.memory_utils as memory_utils

# importing matplotlib modules
from PIL import Image
from skimage.transform import resize

from engine.wavelet import (
    print_size_in_MB,
    to_dense,
    to_sparse,
)

import engine.wavelet as wavelet


def stage_quantization(
    tensor,
    compression_ratio=0.9,
    compression_type="topk",
    device="cuda",
    verbose=0,
    threshold_factor_list=None,
    threshold_prev_list=None,
):
    tensor_new = list()
    energy_list = list()
    threshold_list = list()

    if threshold_factor_list is not None:
        threshold_prev_list_temp = threshold_prev_list.copy()
        threshold_factor_list_temp = threshold_factor_list.copy()

    y = tensor
    if threshold_factor_list is None:
        threshold_prev, threshold_factor = None, None
    else:
        threshold_prev = threshold_prev_list_temp[0]
        threshold_prev_list_temp.pop(0)
        threshold_factor = threshold_factor_list_temp[0]
        threshold_factor_list_temp.pop(0)

    if compression_ratio < 1:
        quantized_tensor, norm, threshold = wavelet.quantize(
            y,
            compression_ratio=compression_ratio,
            n_levels=0,
            compression_type=compression_type,
            device=device,
            threshold_factor=threshold_factor,
            threshold_prev=threshold_prev,
        )

    else:
        quantized_tensor, norm, threshold = (
            torch.zeros_like(y),
            torch.Tensor(0),
            torch.Tensor(0),
        )

    threshold_list.append(threshold)
    energy_list.append(norm)
    tensor_new = quantized_tensor

    if verbose == 1:
        total = memory_utils.get_size_list_tensor_in_gpu(tensor)
        print_size_in_MB("after quantization", total)

    return energy_list, threshold_list, tensor_new


def stage_compress(Y, verbose=0):
    Y_new = to_sparse(Y)

    if verbose == 1:
        total = 0
        total = memory_utils.get_size_list_tensor_in_gpu(Y_new)
        print_size_in_MB("after compress", total)

    return Y_new


def stage_decompress(X, verbose=0):
    Y = to_dense(X)

    if verbose == 1:
        total = memory_utils.get_size_list_tensor_in_gpu(Y)
        print_size_in_MB("after decompress", total)

    return Y


def stage_reconstruct(y, old_shape, device="cuda", verbose=0):
    X_reconstruct = y.view(old_shape)

    if verbose == 1:
        print_size_in_MB(
            "reconstructed", memory_utils.get_size_list_tensor_in_gpu(X_reconstruct)
        )

    return X_reconstruct


def compress(
    tensor,
    compression_ratio=0.9,
    compression_type="topk",
    threshold_factor_list=None,
    threshold_prev_list=None,
    is_footprint=False,
):
    r"""Compress a tensor using wavelet transform

    Args:
        tensor: a tensor to be compressed using wavelet transform
        compression_ratio: compression ratio [0, 1]
        threshold_factor_list: list of threshold factors
        threshold_prev_list: list of previous thresholds
            (The idea of using threshold_factor_list and threshold_prev_list is to
            compute adaptive threshold in quantization stage such that 90 % of energy
            is preserved)

    Returns:
        energy_list: list of current wave energies used in each compressed layer
        threshold_curr_list: list of current thresholds used in each compressed layer

    """

    # quantization stage
    energy_list, threshold_curr_list, tensor = stage_quantization(
        tensor,
        compression_ratio=compression_ratio,
        compression_type=compression_type,
        threshold_factor_list=threshold_factor_list,
        threshold_prev_list=threshold_prev_list,
    )

    if is_footprint:
        size_Y = memory_utils.get_size_tensor_in_gpu(tensor)

    # compression stage
    Yl = stage_compress(tensor)

    if is_footprint:
        return energy_list, threshold_curr_list, Yl, size_Y
    else:
        return energy_list, threshold_curr_list, Yl


def decompress(inp):
    x_sparse, old_shape = inp[0], inp[1]
    y = stage_decompress(x_sparse, verbose=0)
    return stage_reconstruct(y, old_shape)


def start_timer():
    start = time.time()
    return start


def end_timer(start, total_time):
    process_time = (time.time() - start) * 1000
    total_time += process_time
    print("\t\t\t\t\t in {0:.2f} milisecs".format(process_time))
    return total_time


def test_memory_3d(compression_type, compression_ratio, B, C, D):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    print_utils.print_separator()
    if cuda:
        print("GPU | {} | {}".format(compression_type, compression_ratio))
    else:
        print("CPU | {} | {}".format(compression_type, compression_ratio))

    # Load image

    original = np.array(Image.open("database/data/testing/lena.png").convert("LA"))[
        :, :, 0
    ]

    original = resize(original, (128, 128))

    # original = pywt.data.camera()

    fig = plt.figure()
    plt.imshow(original, cmap=plt.cm.gray)
    plt.title("original")

    X = torch.zeros(B, C, 128, 128, D).to(device)

    for i in range(B):
        for j in range(C):
            for k in range(D):
                X[i, j, :, :, k] = torch.from_numpy(original).clone()

    total_time = 0

    start = start_timer()
    total_time = end_timer(start, total_time)

    start = start_timer()
    _, _, y = stage_quantization(
        X,
        compression_ratio=compression_ratio,
        compression_type=compression_type,
        device=device,
        verbose=1,
    )
    total_time = end_timer(start, total_time)

    start = time.time()
    y = stage_compress(y, verbose=1)
    total_time = end_timer(start, total_time)

    start = time.time()
    y = stage_decompress(y, verbose=1)
    total_time = end_timer(start, total_time)

    start = time.time()
    X_reconstruct = stage_reconstruct(y, X.shape, device=device, verbose=1)
    total_time = end_timer(start, total_time)

    print_utils.print_separator()
    print("Total {0:.2f} secs".format(total_time))

    y_slice = X_reconstruct[0, 0, :, :, 0].cpu().numpy()

    fig = plt.figure()
    plt.imshow(y_slice, cmap=plt.cm.gray)
    plt.title("decompresed")

    plt.show()


def test2():
    compression_ratio = 0.01
    compression_type = "topk"

    success = list()
    print("=" * 60)
    test_memory_3d(compression_type, compression_ratio, 1, 3, 128)


def main():
    test2()


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed_time = time.time() - start
    # print("elapsed_time: {0:.2f} [sec]".format(elapsed_time))
