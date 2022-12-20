# 패키지 import
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet18_3D(nn.Module):
    def __init__(self, n_classes):
        super(ResNet18_3D, self).__init__()

        # 파라미터 설정
        block_config = [2, 2, 2, 2]  # resnet18
        img_size = 160
        img_size_8 = 20  # img_size의 1/8로 설정

        # 4개의 모듈을 구성하는 서브 네트워크 준비
        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlock(
            n_blocks=block_config[0], in_channels=64, out_channels=64, stride=1)
        self.feature_res_2 = ResidualBlock(
            n_blocks=block_config[1], in_channels=64, out_channels=128, stride=2)
        self.feature_res_3 = ResidualBlock(
            n_blocks=block_config[2], in_channels=128, out_channels=256, stride=2)
        self.feature_res_4 = ResidualBlock(
            n_blocks=block_config[3], in_channels=256, out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512,n_classes)

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_res_3(x)

        x = self.feature_res_4(x)

        x = self.avgpool(x)
        x=x.view(x.size(0),-1)
        output = self.fc(x)

        return output


class conv3DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(conv3DBatchNormRelu, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding=padding, bias=bias)
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # inplase 설정으로, 입력을 저장하지 않고 출력을 계산하여 메모리 절약

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs


class FeatureMap_convolution(nn.Module):
    def __init__(self):
        '''구성할 네트워크 준비'''
        super(FeatureMap_convolution, self).__init__()

        # 합성곱 층1
        in_channels, out_channels, kernel_size, stride, padding, bias = 3, 64, 7, 2, 3, False
        self.cbnr_1 = conv3DBatchNormRelu(
            in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        # MaxPooling 층
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        outputs = self.maxpool(x)
        return outputs


class ResidualBlock(nn.Sequential):
    def __init__(self, n_blocks, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()

        # BasicBlock를 준비
        self.add_module(
            "block1",
            BasicBlock(in_channels,
                          out_channels, stride)
        )

        # BasicBlock_skip 반복 준비
        for i in range(n_blocks - 1):
            self.add_module(
                "block" + str(i+2),
                BasicBlock(
                    out_channels, out_channels, 1)
            )


class conv3DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(conv3DBatchNorm, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding=padding, bias=bias)
        self.batchnorm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs

class convTranspose2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, bias=True):
        super(convTranspose2DBatchNormRelu, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size, stride, padding, output_padding=output_padding, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = nn.ReLU(inplace=True)(x)
        return outputs


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.cbr_1 = conv3DBatchNormRelu(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.cbr_2 = conv3DBatchNorm(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # 스킵 결합
        if in_channels != out_channels:
            self.cb_residual = conv3DBatchNorm(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
          self.cb_residual = lambda x: x

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_2(self.cbr_1(x))
        residual = self.cb_residual(x)
        result = self.relu(conv + residual)
        return result

class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes, p_dropout=0.1):
        super(DecodePSPFeature, self).__init__()

        # forward에 사용하는 화상 크기
        self.height = height
        self.width = width

        self.cbr = conv3DBatchNormRelu(
            in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=p_dropout)
        self.classification = nn.Conv2d(
            in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output
