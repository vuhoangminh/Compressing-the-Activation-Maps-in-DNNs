import time
import torch
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse
import engine.utils.print_utils as print_utils
import engine.utils.memory_utils as memory_utils
import heapq

# importing matplotlib modules
from PIL import Image
from skimage.transform import resize


def reshape_to_dim_4(x):
    shape = x.shape
    if len(shape) == 1:
        new_shape = (1, 1, 1, shape[0])
    elif len(shape) == 2:
        new_shape = (1, 1, shape[0], shape[1])
    elif len(shape) == 3:
        new_shape = (1, shape[0], shape[1], shape[2])
    elif len(shape) == 4:
        new_shape = shape
    else:
        new_shape = (shape[0], shape[1], shape[2], -1)
    x = x.view(new_shape)
    return x, shape


def print_size_in_MB(string, size_in_B):
    print("Memory {}: {:.3f} MB".format(string, size_in_B / 1024 / 1024))


def get_threshold_topk(
    x, compression_ratio=0.9, n_levels=3, is_numpy=False, is_remove_y_low_low=True
):
    """
    For top-k, we have to remove more k from all sub-images except for the Y_{low, low}
    """
    x = x.view(-1)
    if is_numpy:
        x = x * x
        x = x.cpu().numpy()
        k = int(x.size * (1 - compression_ratio))
        threshold = heapq.nlargest(k, x)[-1]
    else:
        x = torch.abs(x)
        if n_levels == 0:  # for ST
            k_max = x.size()[0] * (1 - compression_ratio)
        else:  # for WT
            if is_remove_y_low_low:
                k_max = x.size()[0] * (1 - compression_ratio)
            else:
                k_max = (
                    x.size()[0] * (1 - compression_ratio) * (1 - (1 / 2**n_levels) ** 2)
                )

        k_max = int(k_max)
        k = max(
            1, k_max
        )  # use max 1 to avoid k=0 when compression ratio converges to 1

        # this method can take 0, thus ineffective for sparse encoding
        # threshold = torch.topk(x, k)[0][-1]

        # this method shouldn't take 0
        index_nonzero = torch.nonzero(torch.topk(x, k)[0])[-1][0]
        threshold = torch.topk(x, k)[0][index_nonzero]

        # threshold = torch.sqrt(threshold)
    return -threshold, threshold, torch.zeros(1), torch.zeros(1)


def get_threshold_est(x, compression_ratio=0.9):
    x = x.view(-1)
    x = x * x
    min_val = torch.min(x)
    max_val = torch.max(x)
    threshold = torch.sqrt(compression_ratio * (max_val - min_val))
    return -threshold, threshold


def quantize(
    x,
    compression_ratio=0.9,
    n_levels=3,
    compression_type="topk",
    device="cuda",
    threshold_factor=None,
    threshold_prev=None,
):
    # =========================================================
    # rerun experiment
    # =========================================================
    compression_type = "topk"
    # =========================================================
    # rerun experiment
    # =========================================================

    if compression_type == "topk":
        threshold_low, threshold_high, norm, threshold = get_threshold_topk(
            x, compression_ratio=compression_ratio, n_levels=n_levels
        )
    else:
        raise ValueError("Compression type is NotImplemented. Please check")

    indices = torch.zeros(x.size())
    indices[x >= threshold_high] = 1
    indices[x <= threshold_low] = 1

    return torch.mul(x, indices.to(device)), norm, threshold


def to_sparse(x):
    """converts dense tensor x to sparse format"""
    x = x.view(-1)
    indices = torch.nonzero(x)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return [indices, values, x.size()]


def to_dense(x):
    indices = x[0]
    values = x[1]
    size = x[2]
    x_typename = torch.typename(values).split(".")[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    return sparse_tensortype(indices, values, size).to_dense()


def get_shape_for_reconstruction(Yl, Yh):
    list_shape = list()
    list_shape.append(Yl.shape)

    for y in Yh:
        list_shape.append(y.shape)
    return list_shape


def reshape_tensors_for_reconstruction(Yl, Yh, list_shape):
    shape = list_shape.pop(0)
    Yh_new = list()

    Yl = Yl.view(shape)
    for count, shape in enumerate(list_shape):
        Yh_new.append(Yh[count].view(shape))
    return Yl, Yh_new


def compute_compression_ratio_energy(x, x_hat, verbose=0):
    norm_x = x * x
    norm_diff = (x - x_hat) * (x - x_hat)
    loss = torch.sum(norm_diff) / torch.sum(norm_x) * 100
    if verbose == 1:
        print("Loss ratio: {0:.2f} %".format(loss.item()))
    return loss.item()


def stage_wavelet_transform(X, w, n_levels=3, device="cuda", verbose=0):
    xfm = DWTForward(J=n_levels, wave=w, mode="symmetric").to(device)

    if verbose == 1:
        print_size_in_MB("original", memory_utils.get_size_list_tensor_in_gpu(X))

    try:
        Yl, Yh = xfm(X)
    except:
        Yl, Yh = xfm(X.type(torch.float32))  # debug float16, 32

    Yl = Yl.type(X.type())
    Yh = [y.type(X.type()) for y in Yh]

    if verbose == 1:
        total = memory_utils.get_size_list_tensor_in_gpu(Yl)
        for yh_item in Yh:
            total += memory_utils.get_size_list_tensor_in_gpu(yh_item)
        print_size_in_MB("after wavelet transform", total)

    list_shape = get_shape_for_reconstruction(Yl, Yh)

    return Yl, Yh, list_shape


def stage_quantization(
    Yl,
    Yh,
    compression_ratio=0.9,
    n_levels=3,
    compression_type="topk",
    is_remove_y_low_low=True,
    device="cuda",
    verbose=0,
    threshold_factor_list=None,
    threshold_prev_list=None,
):
    Yl_new = list()
    Yh_new = list()
    energy_list = list()
    threshold_list = list()

    if threshold_factor_list is not None:
        threshold_prev_list_temp = threshold_prev_list.copy()
        threshold_factor_list_temp = threshold_factor_list.copy()

    # threshold Y_{low,low}
    if is_remove_y_low_low:
        y = Yl
        if threshold_factor_list is None:
            threshold_prev, threshold_factor = None, None
        else:
            threshold_prev = threshold_prev_list_temp[0]
            threshold_prev_list_temp.pop(0)
            threshold_factor = threshold_factor_list_temp[0]
            threshold_factor_list_temp.pop(0)

        if compression_ratio < 1:
            quantized_tensor, norm, threshold = quantize(
                y,
                compression_ratio=compression_ratio,
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
        Yl = quantized_tensor.type(torch.float32)

    else:  # wihout threshold Y_{low,low}
        threshold_list.append(torch.Tensor(0))
        energy_list.append(torch.Tensor(0))
        Yl = Yl.type(torch.float32)

    for y in Yh:
        if threshold_factor_list is None:
            threshold_prev, threshold_factor = None, None
        else:
            threshold_prev = threshold_prev_list_temp[0]
            threshold_prev_list_temp.pop(0)
            threshold_factor = threshold_factor_list_temp[0]
            threshold_factor_list_temp.pop(0)

        if compression_ratio < 1:
            quantized_tensor, norm, threshold = quantize(
                y,
                compression_ratio=compression_ratio,
                n_levels=n_levels,
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

        # else:
        #     quantized_tensor, norm, threshold = y, torch.Tensor(0), torch.Tensor(0)

        threshold_list.append(threshold)
        energy_list.append(norm)

        Yh_new.append(quantized_tensor)
        # Yh_new.append(quantized_tensor.type(y.type()))

    Yh = Yh_new

    if verbose == 1:
        total = memory_utils.get_size_list_tensor_in_gpu(Yl)
        for yh_item in Yh:
            total += memory_utils.get_size_list_tensor_in_gpu(yh_item)
        print_size_in_MB("after quantization", total)

    return energy_list, threshold_list, Yl, Yh


def stage_compress(Yl, Yh, is_remove_y_low_low=True, verbose=0):
    Yl = to_sparse(Yl)
    Yh_new = list()
    for y in Yh:
        Yh_new.append(to_sparse(y))

    Yh = Yh_new

    if verbose == 1:
        total = 0
        total = memory_utils.get_size_list_tensor_in_gpu(Yl)
        for yh_item in Yh:
            total += memory_utils.get_size_list_tensor_in_gpu(yh_item)
        print_size_in_MB("after compress", total)

    return Yl, Yh


def stage_decompress(Yl, Yh, is_remove_y_low_low=True, verbose=0):
    if is_remove_y_low_low:
        Yl = to_dense(Yl)

    Yh_new = list()
    for y in Yh:
        Yh_new.append(to_dense(y))
    Yh = Yh_new

    if verbose == 1:
        total = memory_utils.get_size_list_tensor_in_gpu(Yl)
        for yh_item in Yh:
            total += memory_utils.get_size_list_tensor_in_gpu(yh_item)
        print_size_in_MB("after decompress", total)

    return Yl, Yh


def stage_reconstruct(
    Yl, Yh, list_shape, old_shape, device="cuda", verbose=0, wave="db3"
):
    Yl, Yh = reshape_tensors_for_reconstruction(Yl, Yh, list_shape)

    ifm = DWTInverse(wave=wave, mode="symmetric").to(device)
    X_reconstruct = ifm((Yl, Yh))

    if verbose == 1:
        print_size_in_MB(
            "reconstructed", memory_utils.get_size_list_tensor_in_gpu(X_reconstruct)
        )

    X_reconstruct = X_reconstruct.view(old_shape)
    return X_reconstruct


def compress(
    tensor,
    wave="db3",
    compression_ratio=0.9,
    compression_type="topk",
    n_levels=3,
    is_remove_y_low_low=True,
    threshold_factor_list=None,
    threshold_prev_list=None,
    is_footprint=False,
    verbose=0,
):
    r"""Compress a tensor using wavelet transform

    Args:
        tensor: a tensor to be compressed using wavelet transform
        wave: type of wavelet transform. More detail: pywt
        compression_ratio: compression ratio [0, 1]
        threshold_factor_list: list of threshold factors
        threshold_prev_list: list of previous thresholds
            (The idea of using threshold_factor_list and threshold_prev_list is to
            compute adaptive threshold in quantization stage such that 90 % of energy
            is preserved)

    Returns:
        energy_list: list of current wave energies used in each compressed layer
        threshold_curr_list: list of current thresholds used in each compressed layer
        tuple of (Yl, Yh, list_shape, old_shape) while:
            Yl: low pass filtered output
            Yh: high pass filtered output
            list_shape: list of outputs' shapes for reconstruction
            old_shape: original shape of tensor input
    """

    w = pywt.Wavelet(wave)

    tensor, old_shape = reshape_to_dim_4(tensor)

    # wavele transform stage
    Yl, Yh, list_shape = stage_wavelet_transform(
        tensor, w, n_levels=n_levels, verbose=verbose
    )

    if is_footprint:
        size_Y = memory_utils.get_size_tensor_in_gpu(Yl)
        size_Y += memory_utils.get_size_list_tensor_in_gpu(Yh)

    # quantization stage
    energy_list, threshold_curr_list, Yl, Yh = stage_quantization(
        Yl,
        Yh,
        n_levels=n_levels,
        compression_ratio=compression_ratio,
        compression_type=compression_type,
        threshold_factor_list=threshold_factor_list,
        threshold_prev_list=threshold_prev_list,
        is_remove_y_low_low=is_remove_y_low_low,
        verbose=verbose,
    )

    # compression stage
    Yl, Yh = stage_compress(
        Yl, Yh, is_remove_y_low_low=is_remove_y_low_low, verbose=verbose
    )

    if is_footprint:
        return energy_list, threshold_curr_list, (Yl, Yh, list_shape, old_shape), size_Y
    else:
        return energy_list, threshold_curr_list, (Yl, Yh, list_shape, old_shape)


def decompress(inp, wave="db3", is_remove_y_low_low=True, verbose=0):
    Yl, Yh, list_shape, old_shape = inp[0], inp[1], inp[2], inp[3]
    Yl, Yh = stage_decompress(
        Yl, Yh, is_remove_y_low_low=is_remove_y_low_low, verbose=verbose
    )
    return stage_reconstruct(Yl, Yh, list_shape, old_shape, wave=wave, verbose=verbose)


def test_memory_2d(compression_type, compression_ratio, B, C, wave="db3"):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    w = pywt.Wavelet(wave)

    print_utils.print_separator()
    if cuda:
        print("GPU | {} | {}".format(compression_type, compression_ratio))
    else:
        print("CPU | {} | {}".format(compression_type, compression_ratio))

    # Load image

    original = np.array(Image.open("database/data/testing/cat.jpg").convert("LA"))[
        :, :, 0
    ]

    # original = pywt.data.camera()

    fig = plt.figure()
    plt.imshow(original, cmap=plt.cm.gray)
    plt.title("original")

    X = torch.zeros(B, C, 512, 512).to(device)

    for i in range(B):
        for j in range(C):
            X[i, j, :, :] = torch.from_numpy(original).clone()

    old_shape = X.shape

    start = time.time()
    total_time = 0
    Yl, Yh, list_shape = stage_wavelet_transform(X, w, device=device, verbose=1)
    wavelet_time = time.time() - start
    total_time += wavelet_time
    print("\t\t\t\t\t in {0:.2f} milisecs".format(wavelet_time))

    start = time.time()
    _, _, Yl, Yh = stage_quantization(
        Yl,
        Yh,
        compression_ratio=compression_ratio,
        compression_type=compression_type,
        device=device,
        verbose=1,
    )
    quantization_time = time.time() - start
    total_time += quantization_time
    print("\t\t\t\t\t in {0:.2f} milisecs".format(quantization_time))

    start = time.time()
    Yl, Yh = stage_compress(Yl, Yh, verbose=1)
    compress_time = time.time() - start
    total_time += compress_time
    print("\t\t\t\t\t in {0:.2f} milisecs".format(compress_time))

    start = time.time()
    Yl, Yh = stage_decompress(Yl, Yh, verbose=1)

    decompress_time = time.time() - start
    total_time += decompress_time
    print("\t\t\t\t\t in {0:.2f} milisecs".format(decompress_time))

    start = time.time()
    X_reconstruct = stage_reconstruct(
        Yl, Yh, list_shape, old_shape, wave=wave, device=device, verbose=1
    )

    reconstruct_time = time.time() - start
    total_time += reconstruct_time
    print("\t\t\t\t\t in {0:.2f} milisecs".format(reconstruct_time))

    print_utils.print_separator()
    print("Total {0:.2f} secs".format(total_time))
    compute_compression_ratio_energy(X, X_reconstruct, verbose=1)

    reconstruct = X_reconstruct[0, 0, :, :].cpu().numpy()

    fig = plt.figure()
    plt.imshow(reconstruct, cmap=plt.cm.gray)
    plt.title("reconstruct")

    plt.show()


def test1():
    compression_ratio = 0.9
    compression_type = "topk"
    success = list()

    # test_memory_2d(compression_type, compression_ratio, 8, 3, wave='gaus3')

    for f in pywt.families():
        print(f, pywt.wavelist(family=f))

        if f in ["dmey", "rbio", "bior", "coif", "sym"]:
            for wave in pywt.wavelist(family=f):
                print("=" * 60)
                print(">> processing ", wave)
                print("=" * 60)

                try:
                    test_memory_2d(compression_type, compression_ratio, 8, 3, wave=wave)
                    success.append(wave)
                except:
                    print("{} failed".format(wave))


def start_timer():
    start = time.time()
    return start


def end_timer(start, total_time):
    process_time = (time.time() - start) * 1000
    total_time += process_time
    print("\t\t\t\t\t in {0:.2f} milisecs".format(process_time))
    return total_time


def test_memory_3d(compression_type, compression_ratio, B, C, D, wave="db3"):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    w = pywt.Wavelet(wave)

    print_utils.print_separator()
    if cuda:
        print("GPU | {} | {}".format(compression_type, compression_ratio))
    else:
        print("CPU | {} | {}".format(compression_type, compression_ratio))

    # Load image

    original = np.array(Image.open("database/data/testing/cat.jpg").convert("LA"))[
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

    X, old_shape = reshape_to_dim_4(X)
    total_time = 0

    start = start_timer()
    Yl, Yh, list_shape = stage_wavelet_transform(X, w, device=device, verbose=1)
    total_time = end_timer(start, total_time)

    start = start_timer()
    _, _, Yl, Yh = stage_quantization(
        Yl,
        Yh,
        compression_ratio=compression_ratio,
        compression_type=compression_type,
        device=device,
        verbose=1,
    )
    total_time = end_timer(start, total_time)

    start = time.time()
    Yl, Yh = stage_compress(Yl, Yh, verbose=1)
    total_time = end_timer(start, total_time)

    start = time.time()
    Yl, Yh = stage_decompress(Yl, Yh, verbose=1)
    total_time = end_timer(start, total_time)

    start = time.time()
    X_reconstruct = stage_reconstruct(
        Yl, Yh, list_shape, old_shape, wave=wave, device=device, verbose=1
    )
    total_time = end_timer(start, total_time)

    # Yl, Yh = reshape_tensors_for_reconstruction(Yl, Yh, list_shape)
    # total_time = end_timer(start, total_time)

    # start = time.time()
    # ifm = DWTInverse(wave=wave, mode="symmetric").to(device)
    # X_reconstruct = ifm((Yl, Yh))
    # total_time = end_timer(start, total_time)

    # X_reconstruct = X_reconstruct.view(old_shape)

    print_utils.print_separator()
    print("Total {0:.2f} secs".format(total_time))
    compute_compression_ratio_energy(X.view(old_shape), X_reconstruct, verbose=1)

    reconstruct = X_reconstruct[0, 0, :, :, 0].cpu().numpy()

    fig = plt.figure()
    plt.imshow(reconstruct, cmap=plt.cm.gray)
    plt.title("reconstruct")

    plt.show()


def test2():
    compression_ratio = 0.999
    compression_type = "topk"

    success = list()

    # for wave in ["bior1.1", "bior2.2", "coif3", "db1", "db3", "db5", "rbio1.3", "sym3"]:
    for wave in ["db3"]:
        print("=" * 60)
        print(">> processing ", wave)
        print("=" * 60)

        test_memory_3d(compression_type, compression_ratio, 1, 3, 128, wave=wave)


def main():
    # test1()
    test2()


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed_time = time.time() - start
    # print("elapsed_time: {0:.2f} [sec]".format(elapsed_time))
