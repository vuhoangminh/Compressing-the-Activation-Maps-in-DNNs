import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import engine.utils.print_utils as print_utils
import engine.utils.memory_utils as memory_utils
import itertools


# importing matplotlib modules
from PIL import Image
from skimage.transform import resize


from engine.wavelet import (
    print_size_in_MB,
    to_dense,
    to_sparse,
)


import torch
import torch_dct
import numpy as np
from numpy import r_


from PIL import Image


import engine.utils._dct as _dct


def perform_nxn_dct(im, device="cuda", n_block=8):
    imsize = im.shape
    x_dct = torch.zeros(imsize).to(device)

    # Do 8x8 DCT on image (in-place)
    for i in r_[: imsize[0] : n_block]:
        for j in r_[: imsize[1] : n_block]:
            x_dct[i : (i + n_block), j : (j + n_block)] = _dct.dct(
                im[i : (i + n_block), j : (j + n_block)]
            )

    return x_dct


def perform_all_dct(im, device="cuda"):
    imsize = im.shape
    im = im.to(device)
    x_dct = torch.zeros(imsize).to(device)

    return _dct.dct(im)


def zig_zag(array, n=None):
    """
    Return a new array where only the first n subelements in zig-zag order are kept.
    The remaining elements are set to 0.
    :param array: 2D array_like
    :param n: Keep up to n subelements. Default: all subelements
    :return: The new reduced array.
    """

    shape = array.shape

    assert len(shape) >= 2, "Array must be a 2D array_like"

    if n == None:
        n = shape[0] * shape[1]
    assert (
        0 <= n <= shape[0] * shape[1]
    ), "n must be the number of subelements to return"

    res = torch.zeros_like(array)

    (j, i) = (0, 0)
    direction = "r"  # {'r': right, 'd': down, 'ur': up-right, 'dl': down-left}
    for subel_num in range(1, n + 1):
        res[j][i] = array[j][i]
        if direction == "r":
            i += 1
            if j == shape[0] - 1:
                direction = "ur"
            else:
                direction = "dl"
        elif direction == "dl":
            i -= 1
            j += 1
            if j == shape[0] - 1:
                direction = "r"
            elif i == 0:
                direction = "d"
        elif direction == "d":
            j += 1
            if i == 0:
                direction = "ur"
            else:
                direction = "dl"
        elif direction == "ur":
            i += 1
            j -= 1
            if i == shape[1] - 1:
                direction = "d"
            elif j == 0:
                direction = "r"

    return res


def chunks(l, n):
    """Yield successive n-sized chunks from l"""
    for i in range(0, len(l), int(n)):
        yield l[i : i + int(n)]


def threshold_dct(x_dct, compression_ratio=0.1, type="topk"):
    type = "topk"

    if type == "topk":
        x = torch.abs(x_dct.reshape(-1))
        # x = x_dct.reshape(-1)
        k = max(
            1, int(x.size()[0] * (1 - compression_ratio))
        )  # use max 1 to avoid k=0 when compression ratio converges to 1
        threshold = torch.topk(x, k)[0][-1]
        dct_thresh = x_dct * (abs(x_dct) >= threshold)
    else:
        x = x_dct.contiguous().view(-1)
        x = x * x
        x_sorted, _ = torch.sort(x)
        norm = torch.sum(x_sorted)
        threshold_value_cumsum = norm * compression_ratio
        x_cumsum = torch.cumsum(x_sorted, dim=0)
        idx = torch.nonzero(x_cumsum >= threshold_value_cumsum)
        threshold = torch.sqrt(x_sorted[idx[0]])
        dct_thresh = x_dct * (abs(x_dct) >= threshold)

    return dct_thresh


def perform_nxn_idct(dct_thres, device="cuda", n_blocks=8):
    imsize = dct_thres.shape
    im_dct = torch.zeros(imsize).to(device)

    for i in r_[: imsize[0] : n_blocks]:
        for j in r_[: imsize[1] : n_blocks]:
            im_dct[i : (i + n_blocks), j : (j + n_blocks)] = _dct.idct(
                dct_thres[i : (i + n_blocks), j : (j + n_blocks)]
            )

    return im_dct


def perform_all_idct(dct_thres, device="cuda"):
    imsize = dct_thres.shape
    im_dct = torch.zeros(imsize).to(device)
    return _dct.idct(dct_thres)


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

    # try this https://github.com/mlomnitz/DiffJPEG/blob/master/DiffJPEG.py
    if compression_ratio < 1:
        # y_dct = perform_8x8_dct(y) # best quality
        y_dct = perform_nxn_dct(y, n_block=8)  # definition
        # y_dct = perform_all_dct(y)  # fastest -> memory exp
        # y_dct = torch_dct.dct(y)

        # quantized_tensor = threshold_dct(y_dct, compression_ratio=compression_ratio)
        quantized_tensor = threshold_dct(
            y_dct, compression_ratio=compression_ratio, type=compression_type
        )
        norm, threshold = torch.Tensor(0), compression_ratio * torch.max(y)
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
    y = y.view(old_shape)
    X_reconstruct = perform_nxn_idct(y, device)
    # X_reconstruct = perform_8x8_idct(y, device, n_blocks=16)
    # X_reconstruct = perform_all_idct(y, device)

    if verbose == 1:
        print_size_in_MB(
            "reconstructed", memory_utils.get_size_list_tensor_in_gpu(X_reconstruct)
        )

    return X_reconstruct


def compress(
    tensor,
    compression_ratio=0.9,
    threshold_factor_list=None,
    threshold_prev_list=None,
    compression_type="topk",
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
    # compression_ratio = 0.999
    compression_ratio = 0.1
    compression_type = "topk"

    success = list()
    print("=" * 60)
    test_memory_3d(compression_type, compression_ratio, 1, 3, 128)


def test1():
    x = torch.randn((3, 128, 128, 128))
    X = torch_dct.dct(x)  # DCT-II done through the last dimension
    y = torch_dct.idct(X)  # scaled DCT-III done through the last dimension
    # assrt (torch.abs(x - y)).sum() < 1e-10  # x == y within numerical tolerance
    assert (torch.abs(x - y)).sum() < 100  # x == y within numerical tolerance


def main():
    test2()
    # test1()


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed_time = time.time() - start
    # print("elapsed_time: {0:.2f} [sec]".format(elapsed_time))
