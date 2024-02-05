"""
Author: Feiyang Wang
COMP3055 Machine Learning Coursework

This file contains all the designed CNN networks for Task 3

All the CNNs are tested on RTX 3070 8GB without errors,
for smaller VRAM the training may generate errors, please use CPU mode.

All design principle, detailed graph relationship and size changes
of every CNN here are illustrated in the coursework report.

The Residual Block code used in CNN 3, 4, 5
is adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

The Dense Layer code used in CNN 5
is adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
"""

from collections import OrderedDict
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################################################################
########################################  CNN 1   ################################################
##################################################################################################
class CNN_1(nn.Module):
    # version 1
    # This class defines the simplest CNN as the benchmark
    # The recorded accuracy of this CNN is 76.5% for full dataset and 59.2% for partial (8000 train) dataset
    def __init__(self):
        super(CNN_1, self).__init__()
        # use nn.Sequential to create a series of convolution and ReLU and Pooling behaviours
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3), nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3), nn.ReLU(inplace=True),
        )

        # the fully connected layer
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # run the sequence defined above
        x = self.network(x)
        x = x.reshape(x.shape[0], -1) # reshape to 1D for fc layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # final output
        return x


##################################################################################################
########################################  CNN 2   ################################################
##################################################################################################
class CNN_2(nn.Module):
    # This class defines the improved CNN 2 from CNN 1
    # by adding more conv layers, various conv kernels, and more pooling layers
    # The recorded accuracy of this CNN is 80.13% for full dataset and 63.7% for partial (8000 train) dataset
    # version 2
    def __init__(self):
        super(CNN_2, self).__init__()
        self.network = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, padding=2), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(32768, 4096)
        self.fc2 = nn.Linear(4096, 10)

    def forward(self, x):
        # run the sequence defined above
        x = self.network(x)
        x = x.reshape(x.shape[0], -1) # reshape to 1D for fc layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = self.fc2(x)  # final output
        return x

##################################################################################################
########################################  CNN 3   ################################################
##################################################################################################
def conv3x3(in_channels, out_channels, stride=1):
    # This conv3*3 function defines a basic 3*3 convolution with 1 padding
    # which will be frequently used in the following networks
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    # This class contains a residual block of a typical resnet used as the part of CNN3
    def __init__(self, in_channels, out_channels, stride=1, if_downsample=True):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        # conv -> norm -> ReLU -> conv -> norm
        self.residual_block = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )

        # if the output of last layer does not match the input of current layer, do a downsample to fit
        if if_downsample:
            self.downsample = nn.Sequential(
                    conv3x3(self.in_channels, out_channels, stride=stride), # use this 3*3 conv to do channel change
                    nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # perform downsample if required
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.residual_block(x) # go through a residual block
        out += residual  # add the residual layer, which is either the input or the downsampled input to the output
        out = self.relu(out) # generate the output
        return out

class CNN_3(nn.Module):
    # This class defines the improved CNN 3 from CNN 2
    # by appending residual blocks to the convolutions which is typically like the CNN 2 but fewer layers
    # The recorded accuracy of this CNN is 83.51% for full dataset and 69.3% for partial (8000 train) dataset
    # version 3
    def __init__(self, block= ResidualBlock, layers = [2, 2, 2], num_classes=10):
        super(CNN_3, self).__init__()
        # define the initial convolution layers before the residual blocks
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            # do not pool again to make the input channel large enough for residual layers
        )

        self.in_channels = 256
        # create residual block with 512 output channel and 2 blocks
        self.layer1 = self.make_layer(block, 512, layers[0])
        # create residual block with 1024 output channel and 2 layers and reduce half of the output feature size
        self.layer2 = self.make_layer(block, 1024, layers[1], 2)
        # create residual block with 2048 output channel and 2 layers and reduce half of the output feature size
        self.layer3 = self.make_layer(block, 2048, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, num_classes) # generate the output

    def make_layer(self, block, out_channels, blocks, stride=1):
        # this function creates the layer including multiple blocks
        downsample = False
        # if stride is applied, the output size of this layer will be smaller
        # or if the input channel and output channels are not the same
        # downsample is applied for the input of the layer (first block) to make residual connect work
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = True
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # create the blocks in the residual layer
        self.in_channels = out_channels
        # there is no need to downsample the input for the rest of the block since the size is set consistent in blocks
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers) # return the layer list

    def forward(self, x):
        # go through the first conv layers
        x = self.conv_layers(x)
        # go through each residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        # reshape for fc
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # dropout is added to reduce overfitting
        x = F.dropout(x, 0.5)
        # generate the output
        x = self.fc2(x)

        return x

##################################################################################################
########################################  CNN 4   ################################################
##################################################################################################
class Conv_Res(nn.Module):
    # this is the modified residual block combined with convolutions, which is used as the subnetwork of CNN 4
    def __init__(self, block= ResidualBlock, layers = [2, 2, 2], num_classes=10):
        super(Conv_Res, self).__init__()
        # define the initial convolution layers before the residual blocks, but fewer than CNN 3
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.in_channels = 128
        # create residual block with 128 output channel and 2 blocks
        self.layer1 = self.make_layer(block, 128, layers[0])
        # create residual block with 256 output channel and 2 layers and reduce half of the output feature size
        self.layer2 = self.make_layer(block, 256, layers[1], 2)
        # create residual block with 512 output channel and 2 layers and reduce half of the output feature size
        self.layer3 = self.make_layer(block, 512, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(2)


    def make_layer(self, block, out_channels, blocks, stride=1):
        # this function creates the layer including multiple blocks
        downsample = False
        # if stride is applied, the output size of this layer will be smaller
        # or if the input channel and output channels are not the same
        # downsample is applied for the input of the layer (first block) to make residual connect work
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = True
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # create the blocks in the residual layer
        self.in_channels = out_channels
        # there is no need to downsample the input for the rest of the block since the size is set consistent in blocks
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers) # return the layer list

    def forward(self, x):
        # go through the first conv layers
        x = self.conv_layers(x)
        # go through each residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        return x


class CNN_4(nn.Module):
    # This class defines the improved CNN 4 from CNN 3
    # by applying multiple convolution + residual blocks with different block configurations and concatenate all the results
    # This CNN intends to make the network more adaptive by utilizing its various block configs
    # The recorded accuracy of this CNN is 84.51% for full dataset and 65.6% for partial (8000 train) dataset
    # version 4
    def __init__(self, num_classes=10):
        super(CNN_4, self).__init__()
        # create 5 modified residual subnetworks with different block configs
        self.resnet1 = Conv_Res(ResidualBlock, [2, 2, 2])
        self.resnet2 = Conv_Res(ResidualBlock, [3, 3, 3])
        self.resnet3 = Conv_Res(ResidualBlock, [4, 4, 4])
        self.resnet4 = Conv_Res(ResidualBlock, [5, 5, 5])
        self.resnet5 = Conv_Res(ResidualBlock, [6, 6, 6])

        self.fc1 = nn.Linear(10240, 5120)
        self.fc2 = nn.Linear(5120, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        # go through the modified residual subnetworks
        x_resnet1 = self.resnet1(x)
        x_resnet2 = self.resnet2(x)
        x_resnet3 = self.resnet3(x)
        x_resnet4 = self.resnet4(x)
        x_resnet5 = self.resnet5(x)

        # concatenate all the results
        x_concat = torch.cat([x_resnet1, x_resnet2, x_resnet3, x_resnet4, x_resnet5], dim=1)

        # reshape for fc
        out = x_concat.view(x_concat.size(0), -1)
        out = F.relu(self.fc1(out))
        # dropout is added to reduce overfitting
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        # generate the output
        out = self.fc4(out)
        return out

##################################################################################################
########################################  CNN 5   ################################################
##################################################################################################

class _DenseLayer(nn.Module):
    # this is a single dense layer which will be used as a sub layer for the dense block
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float
    ) -> None:
        super().__init__()
        # norm -> relu -> conv -> norm -> relu -> conv
        self.norm1: nn.BatchNorm2d
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module(
            "conv1", nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )
        self.norm2: nn.BatchNorm2d
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module(
            "conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        # if drop rate is set, perform dropout in the forward function
        self.drop_rate = float(drop_rate)

    def forward(self, input: Tensor) -> Tensor:
        # the dense layer is like a mini CNN, which performs convolutions
        prev_features = input
        new_features0 = torch.cat(prev_features, 1)
        new_features1 = self.conv1(self.relu1(self.norm1(new_features0)))
        new_features2 = self.conv2(self.relu2(self.norm2(new_features1)))
        # perform dropout if required
        if self.drop_rate > 0:
            new_features2 = F.dropout(new_features2, p=self.drop_rate, training=self.training)
        return new_features2

class _DenseBlock(nn.ModuleDict):
    # this is a single dense block which will be used as a sub block for the densenet
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        # go through every layer in this blcok
        for i in range(num_layers):
            # create the dense layer, each layer produces an output of the channel of growth_rate and append to prev input
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            # concat (append) the output of every layer to its input
            # and use the concated results as the input for the next layer
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    # this is a transition layer which transform the given input channels to the output channels using a 1*1 conv
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    # this defines a modified densenet which will be used for CNN 5
    def __init__(
        self,
        growth_rate = 48,
        block_config = (6, 12, 36), # the parameter has been changed, different from typical four blocks
        num_init_features = 96,
        bn_size = 4,
        drop_rate = 0,
    ) -> None:

        super().__init__()
        # define the initial convolution layers before the dense blocks
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7,padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        num_features = num_init_features
        # create every dense block and add its transition layer except the last one
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            # create the dense block
            self.features.add_module("denseblock%d" % (i + 1), block)
            # update the feature for the transition layer
            num_features = num_features + num_layers * growth_rate
            # perform transition unless the last block
            if i != len(block_config) - 1:
                # half the output of the dense block to prevent exploded parameters
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                # update the features
                num_features = num_features // 2
        # normalize in the end
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # set the default (start) parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # go through the dense blocks
        features = self.features(x)
        # generate the output for future use
        out = F.relu(features, inplace=True)
        return out

class Pure_Res(nn.Module):
    # this class is modified from CNN 4 by removing all the convolution before the residual layers
    def __init__(self, block= ResidualBlock, layers = [2, 2, 2]):
        super(Pure_Res, self).__init__()

        self.in_channels = 2112
        # create residual block with 128 output channel and 2 blocks
        self.layer1 = self.make_layer(block, 128, layers[0])
        # create residual block with 256 output channel and 2 layers and reduce half of the output feature size
        self.layer2 = self.make_layer(block, 256, layers[1], 2)
        # create residual block with 512 output channel and 2 layers and reduce half of the output feature size
        self.layer3 = self.make_layer(block, 512, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(2)

    def make_layer(self, block, out_channels, blocks, stride=1):
        # this function creates the layer including multiple blocks
        downsample = False
        # if stride is applied, the output size of this layer will be smaller
        # or if the input channel and output channels are not the same
        # downsample is applied for the input of the layer (first block) to make residual connect work
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = True
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # create the blocks in the residual layer
        self.in_channels = out_channels
        # there is no need to downsample the input for the rest of the block since the size is set consistent in blocks
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers) # return the layer list

    def forward(self, x):
        # go through each residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class CNN_5(nn.Module):
    # This class defines the improved CNN 5 from CNN 4
    # by using the modified densenet to firstly produce an output
    # which will be fed into the residual layers which only have residual blocks.
    #
    # This can make the network not only deep but also efficient since densenet requires significant computational
    # power due to its concatenate input performs.
    #
    # Meanwhile, this CNN still uses the idea of concatenate results from CNN 4
    #
    # The recorded accuracy of this CNN is 85.05% for full dataset and 59.7% for partial (8000 train) dataset
    #
    # The overall accuracy of this CNN trained on the full dataset is the highest among all CNNs
    # The low partial data performance may be due to the deepness of this network would require more dataset to learn.
    # version 5
    def __init__(self, num_classes=10):
        super(CNN_5, self).__init__()
        # create a densenet to firstly produce an intermediate solution
        self.dense = DenseNet()
        # create pure residual layers
        self.resnet1 = Pure_Res(ResidualBlock, [4, 4, 4])
        self.resnet2 = Pure_Res(ResidualBlock, [5, 5, 5])
        self.resnet3 = Pure_Res(ResidualBlock, [6, 6, 6])

        # fc layers
        self.fc1 = nn.Linear(1536, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        # use densenet to firstly process the input
        x = self.dense(x)
        # let the output of densenet go through every resnet seperately
        x_resnet1 = self.resnet1(x)
        x_resnet2 = self.resnet2(x)
        x_resnet3 = self.resnet3(x)

        # concatenate the results from residual blocks
        out = torch.cat([x_resnet1, x_resnet2, x_resnet3], dim=1)

        # reshape to 1D for fc layer
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))

        # dropout is added to reduce overfitting
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        # generate the output
        out = self.fc4(out)
        return out
