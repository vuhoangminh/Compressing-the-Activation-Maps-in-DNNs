import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNetCompressed(nn.Module):
    def __init__(
        self, compression_method, compression_parameters=None, compressed_layers=[]
    ):
        super(MNISTNetCompressed, self).__init__()

        # v1
        # self.conv1 = nn.Conv2d(1, 512, 3, 1)
        # self.conv2 = nn.Conv2d(512, 64, 3, 1)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

        # v2
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 8, 3, 1)
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.compression = compression_method(self, compression_parameters)
        self.compressed_layers = compressed_layers

    def forward(self, x):
        for layer in ["conv1", "relu1", "conv2"]:
            if layer in self.compressed_layers or "all" in self.compressed_layers:
                with torch.autograd.graph.saved_tensors_hooks(
                    self.compression.pack_hook, self.compression.unpack_hook
                ):
                    x = getattr(self, layer)(x)
            else:
                x = getattr(self, layer)(x)

        if "maxpool" in self.compressed_layers or "all" in self.compressed_layers:
            with torch.autograd.graph.saved_tensors_hooks(
                self.compression.pack_hook, self.compression.unpack_hook
            ):
                x = F.max_pool2d(x, 2)
        else:
            x = F.max_pool2d(x, 2)

        if "all" in self.compressed_layers:
            with torch.autograd.graph.saved_tensors_hooks(
                self.compression.pack_hook, self.compression.unpack_hook
            ):
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = F.log_softmax(x, dim=1)
        else:
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)

        return output


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()

        # v1
        # self.conv1 = nn.Conv2d(1, 512, 3, 1)
        # self.conv2 = nn.Conv2d(512, 64, 3, 1)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

        # v2
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 8, 3, 1)
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        # print(output)
        return output


def main():
    return


if __name__ == "__main__":
    main()


"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
              ReLU-2           [-1, 32, 26, 26]               0
            Conv2d-3           [-1, 64, 24, 24]          18,496
         Dropout2d-4           [-1, 64, 12, 12]               0
            Linear-5                  [-1, 128]       1,179,776
              ReLU-6                  [-1, 128]               0
         Dropout2d-7                  [-1, 128]               0
            Linear-8                   [-1, 10]           1,290
================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.68
Params size (MB): 4.58
Estimated Total Size (MB): 5.26
----------------------------------------------------------------


============================================================
without compression
============================================================
Epoch 0 --- Batch 0 --- before forward
/home/minhvu/anaconda3/envs/pytorch-nightly/lib/python3.9/site-packages/torch/cuda/memory.py:392: FutureWarning: torch.cuda.max_memory_cached has been renamed to torch.cuda.max_memory_reserved
  warnings.warn(
/home/minhvu/anaconda3/envs/pytorch-nightly/lib/python3.9/site-packages/torch/cuda/memory.py:384: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  warnings.warn(
------------------------------------------------------------
max_memory_allocated: 5.31MB
max_memory_cached: 22.00MB
memory_allocated: 4.58MB
memory_cached: 22.00MB
------------------------------------------------------------
Epoch 0 --- Batch 0 --- after forward
------------------------------------------------------------
max_memory_allocated: 757.22MB
max_memory_cached: 796.00MB
memory_allocated: 688.38MB
memory_cached: 796.00MB
------------------------------------------------------------
Epoch 0 --- Batch 0 --- after backward
------------------------------------------------------------
max_memory_allocated: 1467.80MB
max_memory_cached: 1912.00MB
memory_allocated: 19.96MB
memory_cached: 166.00MB
------------------------------------------------------------
memory_allocated: 688.38MB
[0.052742719650268555]


----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
              ReLU-2           [-1, 32, 26, 26]               0
            Conv2d-3           [-1, 64, 24, 24]          18,496
         Dropout2d-4           [-1, 64, 12, 12]               0
            Linear-5                  [-1, 128]       1,179,776
              ReLU-6                  [-1, 128]               0
         Dropout2d-7                  [-1, 128]               0
            Linear-8                   [-1, 10]           1,290
================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.68
Params size (MB): 4.58
Estimated Total Size (MB): 5.26
----------------------------------------------------------------


============================================================
with WT compression
============================================================
Epoch 0 --- Batch 0 --- before forward
------------------------------------------------------------
max_memory_allocated: 1467.80MB
max_memory_cached: 1912.00MB
memory_allocated: 4.58MB
memory_cached: 22.00MB
------------------------------------------------------------
Epoch 0 --- Batch 0 --- after forward
------------------------------------------------------------
max_memory_allocated: 1554.73MB
max_memory_cached: 3122.00MB
memory_allocated: 292.57MB
memory_cached: 1410.00MB
------------------------------------------------------------
Epoch 0 --- Batch 0 --- after backward
------------------------------------------------------------
max_memory_allocated: 2172.01MB
max_memory_cached: 3730.00MB
memory_allocated: 20.33MB
memory_cached: 418.00MB
------------------------------------------------------------
memory_allocated: 292.57MB
[2.9701650142669678]


----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
              ReLU-2           [-1, 32, 26, 26]               0
            Conv2d-3           [-1, 64, 24, 24]          18,496
         Dropout2d-4           [-1, 64, 12, 12]               0
            Linear-5                  [-1, 128]       1,179,776
              ReLU-6                  [-1, 128]               0
         Dropout2d-7                  [-1, 128]               0
            Linear-8                   [-1, 10]           1,290
================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.68
Params size (MB): 4.58
Estimated Total Size (MB): 5.26
----------------------------------------------------------------


============================================================
with random compression
============================================================
Epoch 0 --- Batch 0 --- before forward
------------------------------------------------------------
max_memory_allocated: 2172.01MB
max_memory_cached: 3730.00MB
memory_allocated: 13.74MB
memory_cached: 416.00MB
------------------------------------------------------------
Epoch 0 --- Batch 0 --- after forward
------------------------------------------------------------
max_memory_allocated: 2172.01MB
max_memory_cached: 3730.00MB
memory_allocated: 10.80MB
memory_cached: 416.00MB
------------------------------------------------------------
Epoch 0 --- Batch 0 --- after backward
------------------------------------------------------------
max_memory_allocated: 2172.01MB
max_memory_cached: 3730.00MB
memory_allocated: 20.33MB
memory_cached: 168.00MB
------------------------------------------------------------
memory_allocated: 10.80MB
[1.2718801498413086]
"""
