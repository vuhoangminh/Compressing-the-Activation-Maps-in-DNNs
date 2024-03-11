import time
import torch
import torch.nn as nn
from engine.compression import WaveletTransformCompression
import engine.utils.memory_utils as memory_utils
import engine.custom_layers as custom_layers

# torch.cuda.set_per_process_memory_fraction(1.0, 0)


# W, H, D = 64, 64, 64
BS = 1
Channel = 3
W, H, D = 128, 128, 128


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.size(0)
        # log_prob = torch.sigmoid(logits)
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
        dice_score = 2.0 * intersection / ((logits + targets).sum(-1) + self.epsilon)
        # dice_score = 1 - dice_score.sum() / batch_size
        return torch.mean(1.0 - dice_score)


# ========================================================================================
class custom_conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)

        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        # x = self.bn1(x)
        x = custom_layers.RandomBatchNorm.apply(x)
        x = self.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = custom_layers.RandomBatchNorm.apply(x)
        x = self.relu(x)

        return x


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)

        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        for layer in ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"]:
            x = getattr(self, layer)(x)
            # x = x.type(torch.float32)

        # x = self.conv1(inputs)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        return x


class conv_block_compressed(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        compression_method,
        compression_parameters=None,
        compressed_layers=["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)

        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.compression = compression_method(self, compression_parameters)
        self.compressed_layers = compressed_layers

    def forward(self, x):
        for layer in ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"]:
            if layer in self.compressed_layers or "all" in self.compressed_layers:
                with torch.autograd.graph.saved_tensors_hooks(
                    self.compression.pack_hook, self.compression.unpack_hook
                ):
                    # with torch.autograd.graph.save_on_cpu():
                    # try:
                    #     x = getattr(self, layer)(x)  # convert to float32
                    # except:
                    #     x = getattr(self, layer)(
                    #         x.type(torch.float16)
                    #     )  # convert to float16
                    # try:
                    #     x = getattr(self, layer)(
                    #         x.type(torch.float16)
                    #     )  # convert to float16
                    # except:
                    #     x = getattr(self, layer)(
                    #         x.type(torch.float32)
                    #     )  # convert to float32

                    x = getattr(self, layer)(x)
                    # x = x.type(torch.float32)

            else:
                # try:
                #     x = getattr(self, layer)(
                #         x.type(torch.float16)
                #     )  # convert to float16
                # except:
                #     x = getattr(self, layer)(
                #         x.type(torch.float32)
                #     )  # convert to float32

                x = getattr(self, layer)(x)
                # x = x.type(torch.float32)

        return x


class encoder_block(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        is_compressed=False,
        compression_method=WaveletTransformCompression,
        compression_parameters={
            "wave": "db3",
            "compression_ratio": 0.9,
            "n_levels": 3,
        },
        compressed_layers=["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
    ):
        super().__init__()

        if is_compressed:
            self.conv = conv_block_compressed(
                in_c,
                out_c,
                compression_method,
                compression_parameters,
                compressed_layers,
            )
        else:
            self.conv = conv_block(in_c, out_c)

        self.pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        is_compressed=False,
        compression_method=WaveletTransformCompression,
        compression_parameters={
            "wave": "db3",
            "compression_ratio": 0.9,
            "n_levels": 3,
        },
        compressed_layers=["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)

        if is_compressed:
            self.conv = conv_block_compressed(
                out_c + out_c,
                out_c,
                compression_method,
                compression_parameters,
                compressed_layers,
            )
        else:
            self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)

        x = self.conv(x)
        # try:
        #     x = self.conv(x.type(torch.float16))  # convert to float16
        # except:
        #     x = self.conv(x.type(torch.float32))  # convert to float32

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_base_filters=16):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        """ Encoder """
        self.e1 = encoder_block(self.in_channels, n_base_filters)
        self.e2 = encoder_block(n_base_filters, n_base_filters * 2)
        self.e3 = encoder_block(n_base_filters * 2, n_base_filters * 4)
        self.e4 = encoder_block(n_base_filters * 4, n_base_filters * 8)

        """ Bottleneck """
        self.b = conv_block(n_base_filters * 8, n_base_filters * 16)

        """ Decoder """
        self.d1 = decoder_block(n_base_filters * 16, n_base_filters * 8)
        self.d2 = decoder_block(n_base_filters * 8, n_base_filters * 4)
        self.d3 = decoder_block(n_base_filters * 4, n_base_filters * 2)
        self.d4 = decoder_block(n_base_filters * 2, n_base_filters)

        """ Classifier """
        self.outputs = nn.Conv3d(
            n_base_filters, self.out_channels, kernel_size=1, padding=0
        )

    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs


class UNetCompressed(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_base_filters=16,
        is_compressed=[True, False, False, False],
        compression_method=WaveletTransformCompression,
        compression_parameters={
            "wave": "db3",
            "compression_ratio": 0.9,
            "n_levels": 3,
        },
        compressed_layers=["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
        is_compressed_decoder=False,
    ):
        super(UNetCompressed, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_compressed_decoder = is_compressed_decoder

        """ Encoder """
        self.e1 = encoder_block(
            self.in_channels,
            n_base_filters,
            is_compressed[0],
            compression_method,
            compression_parameters,
            compressed_layers,
        )

        self.e2 = encoder_block(
            n_base_filters,
            n_base_filters * 2,
            is_compressed[1],
            compression_method,
            compression_parameters,
            compressed_layers,
        )

        self.e3 = encoder_block(
            n_base_filters * 2,
            n_base_filters * 4,
            is_compressed[2],
            compression_method,
            compression_parameters,
            compressed_layers,
        )

        self.e4 = encoder_block(
            n_base_filters * 4,
            n_base_filters * 8,
            is_compressed[3],
            compression_method,
            compression_parameters,
            compressed_layers,
        )

        """ Bottleneck """
        self.b = conv_block(n_base_filters * 8, n_base_filters * 16)

        # """ Decoder """
        # is_compressed_decoder = True # for memory experiment

        if not is_compressed_decoder:
            self.d1 = decoder_block(n_base_filters * 16, n_base_filters * 8)
            self.d2 = decoder_block(n_base_filters * 8, n_base_filters * 4)
            self.d3 = decoder_block(n_base_filters * 4, n_base_filters * 2)
            self.d4 = decoder_block(n_base_filters * 2, n_base_filters)

        else:
            self.d1 = decoder_block(
                n_base_filters * 16,
                n_base_filters * 8,
                is_compressed[3],
                compression_method,
                compression_parameters,
                compressed_layers,
            )
            self.d2 = decoder_block(
                n_base_filters * 8,
                n_base_filters * 4,
                is_compressed[2],
                compression_method,
                compression_parameters,
                compressed_layers,
            )
            self.d3 = decoder_block(
                n_base_filters * 4,
                n_base_filters * 2,
                is_compressed[1],
                compression_method,
                compression_parameters,
                compressed_layers,
            )
            self.d4 = decoder_block(
                n_base_filters * 2,
                n_base_filters,
                is_compressed[0],
                compression_method,
                compression_parameters,
                compressed_layers,
            )

        """ Decoder with compression"""

        """ Classifier """
        self.outputs = nn.Conv3d(
            n_base_filters, self.out_channels, kernel_size=1, padding=0
        )

    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs


# =================================================================================================


class custom_encoder_block(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
    ):
        super().__init__()
        self.conv = custom_conv_block(in_c, out_c)
        self.pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class custom_decoder_block(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        is_compressed=False,
        compression_method=WaveletTransformCompression,
        compression_parameters={
            "wave": "db3",
            "compression_ratio": 0.9,
            "n_levels": 3,
        },
        compressed_layers=["conv1", "bn1", "relu1", "conv2", "bn2", "relu2"],
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)

        if is_compressed:
            self.conv = conv_block_compressed(
                out_c + out_c,
                out_c,
                compression_method,
                compression_parameters,
                compressed_layers,
            )
        else:
            self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class CustomUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_base_filters=16):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        """ Encoder """
        self.e1 = custom_encoder_block(self.in_channels, n_base_filters)
        self.e2 = encoder_block(n_base_filters, n_base_filters * 2)
        self.e3 = encoder_block(n_base_filters * 2, n_base_filters * 4)
        self.e4 = encoder_block(n_base_filters * 4, n_base_filters * 8)

        """ Bottleneck """
        self.b = conv_block(n_base_filters * 8, n_base_filters * 16)

        """ Decoder """
        self.d1 = decoder_block(n_base_filters * 16, n_base_filters * 8)
        self.d2 = decoder_block(n_base_filters * 8, n_base_filters * 4)
        self.d3 = decoder_block(n_base_filters * 4, n_base_filters * 2)
        self.d4 = decoder_block(n_base_filters * 2, n_base_filters)

        """ Classifier """
        self.outputs = nn.Conv3d(
            n_base_filters, self.out_channels, kernel_size=1, padding=0
        )

    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs


# =================================================================================================


def without_compression(n_epoch, n_batch, verbose=1):
    # no compression
    net = UNet().cuda()

    # summary(net, (3, 128, 128, 128))

    criterion = DiceLoss().cuda()
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

            x = torch.randn(BS, 3, W, H, D).cuda()

            target = torch.randn(BS, W, H, D).type(torch.LongTensor).cuda()

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
    net = UNetCompressed().cuda()
    criterion = DiceLoss().cuda()
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

            x = torch.randn(BS, 3, W, H, D).cuda()

            target = torch.randn(BS, W, H, D).type(torch.LongTensor).cuda()

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


def main():
    # with_compression()
    verbose = 1
    n_epoch = 1
    n_batch = 1
    without_compression(n_epoch, n_batch, verbose)
    with_compression_hook(n_epoch, n_batch, verbose)


if __name__ == "__main__":
    main()
