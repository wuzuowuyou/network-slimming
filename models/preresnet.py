from __future__ import absolute_import
import math
import torch.nn as nn
from .channel_selection import channel_selection


__all__ = ['resnet']

"""
preactivation resnet with bottleneck design.
"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        self.select = channel_selection(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if type(x) == tuple:
            x_input, flops = x
        else:
            x_input = x
            flops = 0
        residual = x_input

        # BN-ReLU-Conv1
        b, c_in, _, _ = x_input.shape
        out = self.bn1(x_input)
        out = self.select(out)
        out = self.relu(out)
        flops += out.numel()/b
        b, c_in, _, _ = out.shape
        out = self.conv1(out)
        _, c_out, h_out, w_out = out.shape
        flops += 3*3*c_in*c_out*h_out*w_out

        # BN-ReLU-Conv2
        out = self.bn2(out)
        out = self.relu(out)
        flops += out.numel()/b
        c_in = c_out
        out = self.conv2(out)
        _, c_out, h_out, w_out = out.shape
        flops += 3*3*c_in*c_out*h_out*w_out

        if self.downsample is not None:
            c = x_input.shape[1]
            residual = self.downsample(x_input)
            _, cr, hr, wr = residual.shape
            flops += c*cr*hr*wr
        out += residual

        return out, flops


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if type(x) == tuple:
            x_input, flops = x
        else:
            x_input = x
            flops = 0
        residual = x_input

        b, c_in, _, _ = x_input.shape
        out = self.bn1(x_input)
        out = self.select(out)
        out = self.relu(out)
        flops += out.numel()/b
        b, c_in, _, _ = out.shape
        out = self.conv1(out)
        _, c_out, h_out, w_out = out.shape
        flops += 1*1*c_in*c_out*h_out*w_out

        out = self.bn2(out)
        out = self.relu(out)
        flops += out.numel()/b
        c_in = c_out
        out = self.conv2(out)
        _, c_out, h_out, w_out = out.shape
        flops += 3*3*c_in*c_out*h_out*w_out

        out = self.bn3(out)
        out = self.relu(out)
        flops += out.numel()/b
        c_in = c_out
        out = self.conv3(out)
        _, c_out, h_out, w_out = out.shape
        flops += 1*1*c_in*c_out*h_out*w_out

        if self.downsample is not None:
            c = x_input.shape[1]
            residual = self.downsample(x_input)
            _, cr, hr, wr = residual.shape
            flops += c*cr*hr*wr
        out += residual

        return out, flops


class resnet(nn.Module):
    def __init__(self, depth=164, dataset='cifar10', cfg=None, block=BasicBlock):
        super(resnet, self).__init__()
        if block == BasicBlock:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
        else:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [
                128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:3*n])
        self.layer2 = self._make_layer(
            block, 32, n, cfg=cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(
            block, 64, n, cfg=cfg[6*n:9*n], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.select = channel_selection(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.num_class = 10
            self.fc = nn.Linear(64 * block.expansion, 10)
        elif dataset == 'cifar100':
            self.num_class = 100
            self.fc = nn.Linear(64 * block.expansion, 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        flops = 0
        b, c_in, _, _ = x.shape
        x = self.conv1(x)
        _, c_out, h_out, w_out = x.shape
        flops += c_in*c_out*h_out*w_out*3*3

        x, flops1 = self.layer1(x)  # 32x32
        x, flops2 = self.layer2(x)  # 16x16
        x, flops3 = self.layer3(x)  # 8x8

        flops += flops1 + flops2 + flops3
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)
        flops += x.numel()/b

        x = self.avgpool(x)
        flops += x.numel()/b
        x = x.view(x.size(0), -1)
        c1 = x.shape[1]
        x = self.fc(x)
        c2 = x.shape[1]
        flops += c1*c2

        return x, flops
