import torch
from torch import nn

from config import ANCHORS, DEVICE, NUM_CLASSES, SCALES


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_num,
            out_num,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU(),
    )


class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels / 2)

        self.layer1 = conv_batch(
            in_channels, reduced_channels, kernel_size=1, padding=0
        )
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        reduced_channels = int(in_channels / 2)

        self.layer1 = conv_batch(
            in_channels, reduced_channels, kernel_size=1, padding=0
        )
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.layer1 = conv_batch(in_channels, 2 * in_channels)
        self.layer2 = nn.Conv2d(2 * in_channels, (num_classes + 5) * 3, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x):
        out = self.layer1(x)
        out = (
            self.layer2(out)
            .reshape(out.shape[0], 3, self.num_classes + 5, out.shape[2], out.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
        return out


class YOLOv3(nn.Module):
    def __init__(self, input_channels=3, num_classes=NUM_CLASSES):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channels
        self.scaled_anchors = (
            torch.tensor(ANCHORS)
            * torch.tensor(SCALES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(DEVICE)

        # Darknet-53
        self.conv1 = conv_batch(input_channels, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(
            block=DarkResidualBlock, in_channels=64, num_blocks=1
        )
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(
            block=DarkResidualBlock, in_channels=128, num_blocks=2
        )
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(
            block=DarkResidualBlock, in_channels=256, num_blocks=8
        )
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(
            block=DarkResidualBlock, in_channels=512, num_blocks=8
        )
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(
            block=DarkResidualBlock, in_channels=1024, num_blocks=4
        )
        # Darknet-53

        self.conv7 = conv_batch(1024, 512, stride=1, kernel_size=1, padding=0)
        self.conv8 = conv_batch(512, 1024, stride=1)

        # #First scale prediction - 5 conv5 blocks inside
        self.residual_block6 = self.make_layer(
            block=ResidualBlock, in_channels=1024, num_blocks=1
        )
        self.conv9 = conv_batch(1024, 512, kernel_size=1, padding=0)
        self.scaleprediction1 = ScalePrediction(512, num_classes)
        # First scale prediction

        self.conv10 = conv_batch(512, 256, stride=1, kernel_size=1, padding=0)

        # Upsample1
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv11 = conv_batch(256 * 3, 256, stride=1, kernel_size=1, padding=0)
        self.conv12 = conv_batch(256, 512, stride=1)

        # Sec scale prediction
        self.residual_block7 = self.make_layer(
            block=ResidualBlock, in_channels=512, num_blocks=1
        )
        self.conv13 = conv_batch(512, 256, kernel_size=1, padding=0)
        self.scaleprediction2 = ScalePrediction(256, num_classes)
        # Sec scale prediction

        self.conv14 = conv_batch(256, 128, stride=1, kernel_size=1, padding=0)

        # Upsample2
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.conv15 = conv_batch(128 * 3, 128, stride=1, kernel_size=1, padding=0)
        self.conv16 = conv_batch(128, 256, stride=1)

        # Third scale prediciton
        self.residual_block8 = self.make_layer(
            block=ResidualBlock, in_channels=256, num_blocks=1
        )
        self.conv17 = conv_batch(256, 128, kernel_size=1, padding=0)
        self.scaleprediction3 = ScalePrediction(128, num_classes)
        # Third scale prediciton

    def forward(self, x):
        outputs = []
        route_connections = []

        # Darnket-53
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(
            out
        )  # After this have to add output, to concatenation
        route_connections.append(out)
        out = self.conv5(out)
        out = self.residual_block4(
            out
        )  # After this have to add output, to concatenation again
        route_connections.append(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        # Daknet-53

        out = self.conv7(out)
        out = self.conv8(out)

        # First scale prediction
        out = self.residual_block6(out)
        out = self.conv9(out)
        out1 = self.scaleprediction1(out)
        outputs.append(out1)
        # First scale prediction

        out = self.conv10(out)

        # Upsumpling1
        out = self.upsample1(out)

        # Concatenation
        out = torch.cat([out, route_connections[-1]], dim=1)
        route_connections.pop()

        out = self.conv11(out)
        out = self.conv12(out)

        # Sec scale prediction
        out = self.residual_block7(out)
        out = self.conv13(out)
        out2 = self.scaleprediction2(out)
        outputs.append(out2)
        # Sec scale predcition

        out = self.conv14(out)

        # Upsumpling2
        out = self.upsample2(out)

        # Concatenation
        out = torch.cat([out, route_connections[-1]], dim=1)
        route_connections.pop()

        out = self.conv15(out)
        out = self.conv16(out)

        # #Third scale prediction
        out = self.residual_block8(out)
        out = self.conv17(out)
        out3 = self.scaleprediction3(out)
        outputs.append(out3)
        # #Third scale predcition

        return outputs

    def make_layer(self, in_channels, num_blocks, block):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)
