# define DepthSeperabelConv2d with depthwise+pointwise
from torch import nn
import torch

class DepthSeperabelConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, stride, 1, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class MobileNetV1(nn.Module):
    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, num_classes=200):
        super().__init__()

        alpha = width_multiplier

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(alpha * 32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(alpha * 32)),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Sequential(
            DepthSeperabelConv2d(int(alpha * 32), int(alpha * 64), 1),
            DepthSeperabelConv2d(int(alpha * 64), int(alpha * 128), 2),
            DepthSeperabelConv2d(int(alpha * 128), int(alpha * 128), 1),
            DepthSeperabelConv2d(int(alpha * 128), int(alpha * 256), 2),
            DepthSeperabelConv2d(int(alpha * 256), int(alpha * 256), 1),
            DepthSeperabelConv2d(int(alpha * 256), int(alpha * 512), 2),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 1024), 2),
            DepthSeperabelConv2d(int(alpha * 1024), int(alpha * 1024), 2)
        )

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(int(alpha * 1024), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x