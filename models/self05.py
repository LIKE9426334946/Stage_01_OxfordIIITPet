# 自己写的UNet+ResNet18网络，把激活函数ReLU替换为自己的激活函数

import torch
import torch.nn as nn
from torch.nn import functional as F


class MyActivation(nn.Module):

    def __init__(self):
        super().__init__()

        # 可学习的斜率和偏置
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):

        # 原始激活函数
        y = torch.where(x <= 0, 0.01 * x, x)

        # 线性映射（仿射变换）
        return self.weight * y + self.bias


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = MyActivation()
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        residual = self.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = identity + residual
        return self.relu(out)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            MyActivation(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            MyActivation(),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SelfModel_05(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            MyActivation(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))

        self.up4 = DecoderBlock(512, 256, 256)
        self.up3 = DecoderBlock(256, 128, 128)
        self.up2 = DecoderBlock(128, 64, 64)
        self.up1 = DecoderBlock(64, 64, 32)
        self.refine = DoubleConv(32, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x_0):
        stem = self.stem(x_0)
        e1 = self.layer1(self.maxpool(stem))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        x = self.layer4(e3)

        x = self.up4(x, e3)
        x = self.up3(x, e2)
        x = self.up2(x, e1)
        x = self.up1(x, stem)
        x = F.interpolate(x, size=x_0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = SelfModel_05(3, 21)

    relu_count = sum(isinstance(m, nn.ReLU) for m in model.modules())

    my_count = sum(isinstance(m, MyActivation) for m in model.modules())

    print("剩余ReLU数量:", relu_count)
    print("自定义激活函数数量:", my_count)
