import torch
from abc import abstractmethod, ABC
import engine.wavelet as wavelet
import engine.threshold as threshold
import engine.dct as dct


class ConditionalContext:
    def __init__(self, condition, compression):
        self.condition = condition
        self.compression = compression

    def __enter__(self):
        if self.condition:
            return torch.autograd.graph.saved_tensors_hooks(
                self.compression.pack_hook, self.compression.unpack_hook
            ).__enter__()

    def __exit__(self, type, value, traceback):
        if self.condition:
            return torch.autograd.graph.saved_tensors_hooks(
                self.compression.pack_hook, self.compression.unpack_hook
            ).__exit__(type, value, traceback)


def get_compression_params(options):
    if options["model"]["compression_method"] == "wt":
        compression_method = WaveletTransformCompression
        compression_parameters = {
            "wave": options["model"]["wave"],
            "compression_ratio": options["model"]["compression_ratio"],
            "n_levels": options["model"]["n_levels"],
        }
    elif options["model"]["compression_method"] == "th":
        compression_method = ThresholdingCompression
        compression_parameters = {
            "compression_ratio": options["model"]["compression_ratio"],
        }
    elif options["model"]["compression_method"] == "dct":
        compression_method = DCTCompression
        compression_parameters = {
            "compression_ratio": options["model"]["compression_ratio"],
        }
    elif options["model"]["compression_method"] == "random":
        compression_method = RandomCompression
        compression_parameters = None
    else:
        m = options["model"]["compression_method"]
        raise ValueError(f"Compression {m} is NotImplemented. Please check")

    if options["data"]["dataset"] in ["cifar10", "cifar100", "spleen", "brats"]:
        compressed_blocks = options["model"]["compressed_blocks"].split("-")
        return compression_method, compression_parameters, compressed_blocks
    elif options["data"]["dataset"] == "mnist":
        compressed_layers = options["model"]["compressed_layers"].split("-")
        return compression_method, compression_parameters, compressed_layers
    else:
        raise ValueError(f"Dataset is NotImplemented. Please check")


class Compression(ABC):
    @abstractmethod
    def pack_hook():
        pass

    @abstractmethod
    def unpack_hook():
        pass


class NoCompression(Compression):
    """
    input: tensor
    output: input tensor
    """

    def pack_hook(self, x):
        x

    def unpack_hook(self, x):
        return x


class RandomCompression(Compression):
    """
    input: tensor
    output: a random tensor with input tensor's shape
    """

    def __init__(self, net=None, compression_parameters=None):
        super(RandomCompression, self).__init__()
        self.net = net
        self.compression_parameters = compression_parameters

    def pack_hook(self, tensor):
        # print("Packing")
        return tensor.size(), tensor.type()

    def unpack_hook(self, packed):
        # print("Unpacking")
        s, t = packed
        return torch.rand(s).type(t).to("cuda")


class ThresholdingCompression(Compression):
    def __init__(
        self,
        net,
        compression_parameters={
            "compression_ratio": 0.9,
        },
    ):
        super(ThresholdingCompression, self).__init__()
        self.net = net
        self.compression_parameters = compression_parameters
        self.compression_ratio = self.compression_parameters["compression_ratio"]

    def pack_hook(
        self,
        x,
    ):
        is_parameter = False
        for name, w in self.net.named_parameters():
            if sorted(list(w.shape)) == sorted(list(x.shape)) and torch.equal(
                w.data, x.data
            ):
                is_parameter = True
        if not x.requires_grad:
            is_parameter = True

        if is_parameter:
            return x, is_parameter, x.shape, x.type()
        else:
            # print("Packing")
            (
                energy_list,
                threshold_curr_list,
                compressed_input_tuple,
            ) = threshold.compress(
                x,
                compression_ratio=self.compression_ratio,
            )
            return compressed_input_tuple, is_parameter, x.shape, x.type()

    def unpack_hook(self, packed):
        x, is_parameter, old_shape, t = packed
        if is_parameter:
            return x
        else:
            # print("Unpacking")
            x = threshold.decompress([x, old_shape])
            return x.type(t)


class DCTCompression(Compression):
    def __init__(
        self,
        net,
        compression_parameters={
            "compression_ratio": 0.9,
        },
    ):
        super(DCTCompression, self).__init__()
        self.net = net
        self.compression_parameters = compression_parameters
        self.compression_ratio = self.compression_parameters["compression_ratio"]

    def pack_hook(
        self,
        x,
    ):
        is_parameter = False
        for name, w in self.net.named_parameters():
            if sorted(list(w.shape)) == sorted(list(x.shape)) and torch.equal(
                w.data, x.data
            ):
                is_parameter = True
        if not x.requires_grad:
            is_parameter = True

        if is_parameter:
            return x, is_parameter, x.shape, x.type()
        else:
            # print("Packing")
            (
                energy_list,
                threshold_curr_list,
                compressed_input_tuple,
            ) = dct.compress(
                x,
                compression_ratio=self.compression_ratio,
            )
            return compressed_input_tuple, is_parameter, x.shape, x.type()

    def unpack_hook(self, packed):
        x, is_parameter, old_shape, t = packed
        if is_parameter:
            return x
        else:
            # print("Unpacking")
            x = dct.decompress([x, old_shape])
            return x.type(t)


class WaveletTransformCompression(Compression):
    def __init__(
        self,
        net,
        compression_parameters={
            "wave": "db3",
            "compression_ratio": 0.9,
            "n_levels": 3,
        },
    ):
        super(WaveletTransformCompression, self).__init__()
        self.net = net
        self.compression_parameters = compression_parameters
        self.wave = self.compression_parameters["wave"]
        self.compression_ratio = self.compression_parameters["compression_ratio"]
        self.n_levels = self.compression_parameters["n_levels"]

    def pack_hook(
        self,
        x,
    ):
        is_parameter = False
        for name, w in self.net.named_parameters():
            if sorted(list(w.shape)) == sorted(list(x.shape)) and (
                torch.equal(w.data, x.data)
                or torch.equal(w.type(torch.float16).data, x.type(torch.float16).data)
            ):
                is_parameter = True
        if not x.requires_grad:
            is_parameter = True

        if is_parameter:
            return x, is_parameter, x.type()
        else:
            # print("Packing")
            (
                energy_list,
                threshold_curr_list,
                compressed_input_tuple,
            ) = wavelet.compress(
                x,
                wave=self.wave,
                compression_ratio=self.compression_ratio,
                n_levels=self.n_levels,
            )
            return compressed_input_tuple, is_parameter, x.type()

    def unpack_hook(self, packed):
        x, is_parameter, t = packed
        if is_parameter:
            return x.type(t)
        else:
            # print("Unpacking")
            x = wavelet.decompress(x, wave=self.wave)
            return x.type(t)


def test():
    x = torch.randn(1, 3, 100, 100).cuda()
    (
        energy_list,
        threshold_curr_list,
        compressed_input_tuple,
    ) = wavelet.compress(
        x,
        wave="db3",
        compression_ratio=0,
        n_levels=3,
    )
    x_wt = wavelet.decompress(compressed_input_tuple, wave="db3")
    print(torch.all(x.eq(x_wt)))

    (
        energy_list,
        threshold_curr_list,
        compressed_input_tuple,
    ) = threshold.compress(
        x,
        compression_ratio=0,
    )
    old_shape = x.shape
    x_th = threshold.decompress([compressed_input_tuple, old_shape])
    print(torch.all(x.eq(x_th)))

    (
        energy_list,
        threshold_curr_list,
        compressed_input_tuple,
    ) = dct.compress(
        x,
        compression_ratio=0,
    )
    x_dct = dct.decompress([compressed_input_tuple, old_shape])
    print(torch.all(x.eq(x_dct)))

    a = 2


if __name__ == "__main__":
    # main()
    test()
