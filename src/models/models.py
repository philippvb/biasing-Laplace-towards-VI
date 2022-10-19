from typing import List
from torch import nn
from math import floor
import torch
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator
import math
from torch.nn import functional as F
import argparse
from backpack.custom_module.branching import Parallel, SumModule

class SmallCNN(nn.Sequential):
    def __init__(self, channel_sizes, image_dims, kernel_size=3, n_classes=10, *args, **kwargs) -> None:
        model_list = []
        padding = floor(kernel_size/2)
        channel_sizes = [image_dims[0]] + channel_sizes
        for in_size, out_size in zip(channel_sizes[:-1], channel_sizes[1:]):
            model_list.append(nn.Conv2d(in_size, out_size, kernel_size, 1, padding))
            model_list.append(nn.MaxPool2d(2))
            model_list.append(nn.ReLU())
        model_list.append(nn.Flatten())
        super().__init__(*model_list)

        # test forward pass to determine dims
        test_tensor = torch.rand([1] + list(image_dims))
        with torch.no_grad():
            out = self(test_tensor)
            dims = len(out.flatten())

        self.add_module("fc_layer", nn.Linear(dims, n_classes))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SmallCnn")
        parser.add_argument("--channel_sizes", type=List[int], default=[8, 16, 32])
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--n_classes", type=int, default=10)
        return parent_parser

@variational_estimator
class VariationalSmallCNN(nn.Sequential):
    def __init__(self, channel_sizes, image_dims, kernel_size=3, n_classes=10, *args, **kwargs) -> None:
        model_list = []
        padding = floor(kernel_size/2)
        channel_sizes = [image_dims[0]] + channel_sizes
        for in_size, out_size in zip(channel_sizes[:-1], channel_sizes[1:]):
            model_list.append(BayesianConv2d(in_size, out_size, kernel_size=[kernel_size, kernel_size], stride=1, padding=padding))
            model_list.append(nn.MaxPool2d(2))
            model_list.append(nn.ReLU())
        model_list.append(nn.Flatten())
        super().__init__(*model_list)

        # test forward pass to determine dims
        test_tensor = torch.rand([1] + list(image_dims))
        with torch.no_grad():
            out = self(test_tensor)
            dims = len(out.flatten())

        self.add_module("fc_layer", BayesianLinear(dims, n_classes))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VariationalSmallCNN")
        parser.add_argument("--channel_sizes", type=List[int], default=[8, 16, 32])
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--n_classes", type=int, default=10)
        return parent_parser


class MLP(nn.Sequential):
    def __init__(self, layer_sizes) -> None:
        model_list = [nn.Flatten()]
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            model_list.append(nn.Linear(in_dim, out_dim))
            model_list.append(nn.ReLU())
        super().__init__(*model_list[:-1])

@variational_estimator
class VariationalMLP(nn.Sequential):
    def __init__(self, layer_sizes) -> None:
        model_list = [nn.Flatten()]
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            model_list.append(BayesianLinear(in_dim, out_dim))
            model_list.append(nn.ReLU())
        super().__init__(*model_list[:-1])


class ToyModel2D(nn.Sequential):
    def __init__(self, **kwargs):
        h = 50  # num. hidden units per layer
        n = 1
        feature_map = nn.Sequential(
            nn.Linear(n, h),
            nn.Tanh()
        )
        clf = nn.Linear(h, 1, bias=True)
        super(ToyModel2D, self).__init__(feature_map, clf)

@variational_estimator
class VariationalToyModel2D(nn.Sequential):
    def __init__(self, **kwargs):
        h = 20  # num. hidden units per layer
        n = 1
        feature_map = nn.Sequential(
            BayesianLinear(n, h),
            nn.Tanh()
        )
        clf = BayesianLinear(h, 1, bias=True)
        super(VariationalToyModel2D, self).__init__(feature_map, clf)


class BasicBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        if not (in_planes == out_planes):
            skip_connection = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
            first_layers = [
                nn.BatchNorm2d(in_planes),
                nn.ReLU()]

            residual_layers = [
                    nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU()]
        else:
            skip_connection = nn.Identity()
            first_layers = [nn.Identity()]
            residual_layers = [
                    nn.BatchNorm2d(in_planes),
                    nn.ReLU(),
                    nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU()]

        if dropRate > 0:
            residual_layers.append(nn.Dropout(p=dropRate))
        # second conv layer
        residual_layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False))


        layers = [
            nn.Sequential(*first_layers),
            Parallel(
                nn.Sequential(skip_connection),
                nn.Sequential(*residual_layers),
                merge_module=SumModule()
            )
        ]

        super(BasicBlock, self).__init__(*layers)



class NetworkBlock(nn.Sequential):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
        super(NetworkBlock, self).__init__(*layer)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []

        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))

        return layers

class WideResNet(nn.Sequential):

    def __init__(self, depth, widen_factor, n_classes=10, num_channel=3, dropRate=0.3, feature_extractor=False, **kwargs):

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        layers = [nn.Conv2d(num_channel, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False),
        NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate),
        NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate),
        NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate),
        nn.BatchNorm2d(nChannels[3]),
        nn.ReLU(),
        nn.AvgPool2d(8),
        nn.Flatten(),
        ]
        if not feature_extractor:
            layers.append(nn.Linear(nChannels[3], n_classes))
        super(WideResNet, self).__init__(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WideResNet")
        parser.add_argument("--depth", type=int) # 16
        parser.add_argument("--widen_factor", type=int) #4
        parser.add_argument("--n_classes", type=int, default=10)
        parser.add_argument("--num_channel", type=int, default=3)
        parser.add_argument("--dropRate", type=float, default=0.3)
        parser.add_argument('--feature_extractor', action=argparse.BooleanOptionalAction, default=False)
        return parent_parser

class WideResNetLastLayer(nn.Module):

    def __init__(self, depth, widen_factor, n_classes=10, num_channel=3, dropRate=0.3, **kwargs):

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        layers = [nn.Conv2d(num_channel, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False),
        NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate),
        NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate),
        NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate),
        nn.BatchNorm2d(nChannels[3]),
        nn.ReLU(),
        nn.AvgPool2d(8),
        nn.Flatten(),
        ]
        super(WideResNetLastLayer, self).__init__()
        self.feature_extractor = nn.Sequential(*layers)
        self.linear = nn.Linear(nChannels[3], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.linear(self.feature_extractor(x))


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WideResNet")
        parser.add_argument("--depth", type=int) # 16
        parser.add_argument("--widen_factor", type=int) #4
        parser.add_argument("--n_classes", type=int, default=10)
        parser.add_argument("--num_channel", type=int, default=3)
        parser.add_argument("--dropRate", type=float, default=0.3)
        parser.add_argument('--feature_extractor', action=argparse.BooleanOptionalAction, default=False)
        return parent_parser


class VariationalBasicBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        if not (in_planes == out_planes):
            skip_connection = BayesianConv2d(in_planes, out_planes, kernel_size=[1, 1], stride=stride, padding=0, bias=False)
            first_layers = [
                nn.BatchNorm2d(in_planes),
                nn.ReLU()]

            residual_layers = [
                    BayesianConv2d(in_planes, out_planes, kernel_size=[3, 3], stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU()]
        else:
            skip_connection = nn.Identity()
            first_layers = [nn.Identity()]
            residual_layers = [
                    nn.BatchNorm2d(in_planes),
                    nn.ReLU(),
                    BayesianConv2d(in_planes, out_planes, kernel_size=[3, 3], stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU()]

        if dropRate > 0:
            # raise NotImplementedError("Not supported")
            residual_layers.append(nn.Dropout(p=dropRate))
        # second conv layer
        residual_layers.append(BayesianConv2d(out_planes, out_planes, kernel_size=[3, 3], stride=1, padding=1, bias=False))


        layers = [
            nn.Sequential(*first_layers),
            Parallel(
                nn.Sequential(skip_connection),
                nn.Sequential(*residual_layers),
                merge_module=SumModule()
            )
        ]

        super(VariationalBasicBlock, self).__init__(*layers)



@variational_estimator
class VariationalWideResNet(nn.Sequential):

    def __init__(self, depth, widen_factor, n_classes=10, num_channel=3, dropRate=0.3, feature_extractor=False, **kwargs):

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = VariationalBasicBlock

        layers = [BayesianConv2d(num_channel, nChannels[0], kernel_size=[3,3], stride=1, padding=1, bias=False),
        NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate),
        NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate),
        NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate),
        nn.BatchNorm2d(nChannels[3]),
        nn.ReLU(),
        nn.AvgPool2d(8),
        nn.Flatten(),
        ]
        if not feature_extractor:
            layers.append(BayesianLinear(nChannels[3], n_classes))
        super(VariationalWideResNet, self).__init__(*layers)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WideResNet")
        parser.add_argument("--depth", type=int) # 16
        parser.add_argument("--widen_factor", type=int) #4
        parser.add_argument("--n_classes", type=int, default=10)
        parser.add_argument("--num_channel", type=int, default=3)
        parser.add_argument("--dropRate", type=float, default=0.3)
        parser.add_argument('--feature_extractor', action=argparse.BooleanOptionalAction, default=False)
        return parent_parser


