from torch import nn
import torch
import math


def conv3x3_encoder(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_decoder(in_planes, out_planes, stride=1, output_padding=0):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              output_padding=output_padding, padding=1, bias=False)


class BasicBlock_encoder(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_encoder, self).__init__()
        self.conv1 = conv3x3_encoder(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_encoder(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_decoder(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, output_padding=0, upsample=None):
        super(BasicBlock_decoder, self).__init__()
        self.conv1 = conv3x3_decoder(inplanes, planes, stride, output_padding)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_decoder(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class joint_model_1(nn.Module):

    def __init__(self, block_encoder, block_decoder, layers):
        self.inplanes = 64
        super(joint_model_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder_layer1 = self._make_encoder_layer(block_encoder, 64, layers[0])

        self.encoder_layer2 = self._make_encoder_layer(block_encoder, 64, layers[1], stride=2)
        self.encoder_layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        self.encoder_layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)
        self.decoder_layer1 = self._make_decoder_layer(block_decoder, 512, layers[3])
        self.decoder_layer2 = self._make_decoder_layer(block_decoder, 256, layers[2], stride=2, output_padding=1)
        self.decoder_layer3 = self._make_decoder_layer(block_decoder, 64, layers[1], stride=2, output_padding=1)
        self.decoder_layer4 = self._make_decoder_layer(block_decoder, 64, layers[0], stride=2, output_padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=False)

        self.layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)
        self.layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        self.layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=4)
        self.fc = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_encoder_layer(self, block_encoder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_encoder.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_encoder.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_encoder.expansion),
            )

        layers = []
        layers.append(block_encoder(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_encoder.expansion
        for i in range(1, blocks):
            layers.append(block_encoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_decoder_layer(self, block_decoder, planes, blocks, stride=1, output_padding=0):
        upsample = None
        if stride != 1 or self.inplanes != planes * block_decoder.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block_decoder.expansion, kernel_size=1,
                                   output_padding=output_padding, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_decoder.expansion),
            )

        layers = []
        layers.append(block_decoder(self.inplanes, planes, stride, output_padding, upsample))
        self.inplanes = planes * block_decoder.expansion
        for i in range(1, blocks):
            layers.append(block_decoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.encoder_layer1(x)

        # classification part
        out_1 = self.layer2(x)
        out_1 = self.layer3(out_1)
        out_1 = self.layer4(out_1)

        out_1 = self.avgpool(out_1)
        out_1 = out_1.view(out_1.size(0), -1)
        out_1 = self.fc(out_1)

        # auto encoder part
        out_encoder_2 = self.encoder_layer2(x)
        out_encoder_3 = self.encoder_layer3(out_encoder_2)
        out_encoder_4 = self.encoder_layer4(out_encoder_3)
        out_decoder_1 = self.decoder_layer1(out_encoder_4)
        out_decoder_2 = self.decoder_layer2(out_decoder_1)
        out_decoder_3 = self.decoder_layer3(out_decoder_2)
        out_decoder_4 = self.decoder_layer4(out_decoder_3)

        out_2 = self.conv2(out_decoder_4)

        out_3 = self.conv2(out_encoder_2)
        out_4 = self.conv2(out_decoder_3)

        return out_1, out_2, out_3, out_4
        # return out_1


class joint_model_2(nn.Module):

    def __init__(self, block_encoder, block_decoder, layers):
        self.inplanes = 64
        super(joint_model_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder_layer1 = self._make_encoder_layer(block_encoder, 64, layers[0])

        self.encoder_layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)
        self.encoder_layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        # self.encoder_layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)
        # self.decoder_layer1 = self._make_decoder_layer(block_decoder, 512, layers[3])
        # self.decoder_layer2 = self._make_decoder_layer(block_decoder, 256, layers[2], stride=2, output_padding=1)
        self.decoder_layer2 = self._make_decoder_layer(block_decoder, 256, layers[2])
        self.decoder_layer3 = self._make_decoder_layer(block_decoder, 128, layers[1], stride=2, output_padding=1)
        self.decoder_layer4 = self._make_decoder_layer(block_decoder, 64, layers[0], stride=2, output_padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False)

        self.layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)
        self.layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        self.layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(4, stride=4)
        self.fc = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_encoder_layer(self, block_encoder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_encoder.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_encoder.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_encoder.expansion),
            )

        layers = []
        layers.append(block_encoder(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_encoder.expansion
        for i in range(1, blocks):
            layers.append(block_encoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_decoder_layer(self, block_decoder, planes, blocks, stride=1, output_padding=0):
        upsample = None
        if stride != 1 or self.inplanes != planes * block_decoder.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block_decoder.expansion, kernel_size=1,
                                   output_padding=output_padding, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_decoder.expansion),
            )

        layers = []
        layers.append(block_decoder(self.inplanes, planes, stride, output_padding, upsample))
        self.inplanes = planes * block_decoder.expansion
        for i in range(1, blocks):
            layers.append(block_decoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.encoder_layer1(x)

        # classification part
        out_1 = self.layer2(x)
        out_1 = self.layer3(out_1)
        out_1 = self.layer4(out_1)

        out_1 = self.avgpool(out_1)
        out_1 = out_1.view(out_1.size(0), -1)
        out_1 = self.fc(out_1)

        # auto encoder part
        out_2 = self.encoder_layer2(x)
        out_2 = self.encoder_layer3(out_2)
        # out_2 = self.encoder_layer4(out_2)
        # out_2 = self.decoder_layer1(out_2)
        out_2 = self.decoder_layer2(out_2)
        out_2 = self.decoder_layer3(out_2)
        out_2 = self.decoder_layer4(out_2)

        out_2 = self.conv2(out_2)

        return out_1, out_2


class joint_model_3(nn.Module):

    def __init__(self, block_encoder, block_decoder, layers):
        self.inplanes = 64
        super(joint_model_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder_layer1 = self._make_encoder_layer(block_encoder, 64, layers[0])

        self.encoder_layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)
        # self.encoder_layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        # self.encoder_layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)
        # self.decoder_layer1 = self._make_decoder_layer(block_decoder, 512, layers[3])
        # self.decoder_layer2 = self._make_decoder_layer(block_decoder, 256, layers[2], stride=2, output_padding=1)
        # self.decoder_layer3 = self._make_decoder_layer(block_decoder, 128, layers[1], stride=2, output_padding=1)
        self.decoder_layer3 = self._make_decoder_layer(block_decoder, 128, layers[1])
        self.decoder_layer4 = self._make_decoder_layer(block_decoder, 64, layers[0], stride=2, output_padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False)

        self.layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)
        self.layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        self.layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(4, stride=4)
        self.fc = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_encoder_layer(self, block_encoder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_encoder.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_encoder.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_encoder.expansion),
            )

        layers = []
        layers.append(block_encoder(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_encoder.expansion
        for i in range(1, blocks):
            layers.append(block_encoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_decoder_layer(self, block_decoder, planes, blocks, stride=1, output_padding=0):
        upsample = None
        if stride != 1 or self.inplanes != planes * block_decoder.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block_decoder.expansion, kernel_size=1,
                                   output_padding=output_padding, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_decoder.expansion),
            )

        layers = []
        layers.append(block_decoder(self.inplanes, planes, stride, output_padding, upsample))
        self.inplanes = planes * block_decoder.expansion
        for i in range(1, blocks):
            layers.append(block_decoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.encoder_layer1(x)

        # classification part
        out_1 = self.layer2(x)
        out_1 = self.layer3(out_1)
        out_1 = self.layer4(out_1)

        out_1 = self.avgpool(out_1)
        out_1 = out_1.view(out_1.size(0), -1)
        out_1 = self.fc(out_1)

        # auto encoder part
        out_2 = self.encoder_layer2(x)
        # out_2 = self.encoder_layer3(out_2)
        # out_2 = self.encoder_layer4(out_2)
        # out_2 = self.decoder_layer1(out_2)
        # out_2 = self.decoder_layer2(out_2)
        out_2 = self.decoder_layer3(out_2)
        out_2 = self.decoder_layer4(out_2)

        out_2 = self.conv2(out_2)

        return out_1, out_2


class joint_model_block2_1(nn.Module):

    def __init__(self, block_encoder, block_decoder, layers):
        self.inplanes = 64
        super(joint_model_block2_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder_layer1 = self._make_encoder_layer(block_encoder, 64, layers[0])
        self.encoder_layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)

        self.encoder_layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        self.encoder_layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)
        self.decoder_layer1 = self._make_decoder_layer(block_decoder, 512, layers[3])
        self.decoder_layer2 = self._make_decoder_layer(block_decoder, 256, layers[2], stride=2, output_padding=1)
        self.decoder_layer3 = self._make_decoder_layer(block_decoder, 128, layers[1], stride=2, output_padding=1)
        self.decoder_layer4 = self._make_decoder_layer(block_decoder, 64, layers[0], stride=2, output_padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False)

        self.inplanes = 128
        # self.layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)
        self.layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        self.layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=4)
        self.fc = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_encoder_layer(self, block_encoder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_encoder.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_encoder.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_encoder.expansion),
            )

        layers = []
        layers.append(block_encoder(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_encoder.expansion
        for i in range(1, blocks):
            layers.append(block_encoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_decoder_layer(self, block_decoder, planes, blocks, stride=1, output_padding=0):
        upsample = None
        if stride != 1 or self.inplanes != planes * block_decoder.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block_decoder.expansion, kernel_size=1,
                                   output_padding=output_padding, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_decoder.expansion),
            )

        layers = []
        layers.append(block_decoder(self.inplanes, planes, stride, output_padding, upsample))
        self.inplanes = planes * block_decoder.expansion
        for i in range(1, blocks):
            layers.append(block_decoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)

        # classification part
        out_1 = self.layer3(x)
        out_1 = self.layer4(out_1)

        out_1 = self.avgpool(out_1)
        out_1 = out_1.view(out_1.size(0), -1)
        out_1 = self.fc(out_1)

        # auto encoder part
        out_2 = self.encoder_layer3(x)
        out_2 = self.encoder_layer4(out_2)
        out_2 = self.decoder_layer1(out_2)
        out_2 = self.decoder_layer2(out_2)
        out_2 = self.decoder_layer3(out_2)
        out_2 = self.decoder_layer4(out_2)

        out_2 = self.conv2(out_2)

        return out_1, out_2


class joint_model_block3_1(nn.Module):

    def __init__(self, block_encoder, block_decoder, layers):
        self.inplanes = 64
        super(joint_model_block3_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder_layer1 = self._make_encoder_layer(block_encoder, 64, layers[0])
        self.encoder_layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)
        self.encoder_layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)

        self.encoder_layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)
        self.decoder_layer1 = self._make_decoder_layer(block_decoder, 512, layers[3])
        self.decoder_layer2 = self._make_decoder_layer(block_decoder, 256, layers[2], stride=2, output_padding=1)
        self.decoder_layer3 = self._make_decoder_layer(block_decoder, 128, layers[1], stride=2, output_padding=1)
        self.decoder_layer4 = self._make_decoder_layer(block_decoder, 64, layers[0], stride=2, output_padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False)

        self.inplanes = 256
        # self.layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)
        # self.layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        self.layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=4)
        self.fc = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_encoder_layer(self, block_encoder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_encoder.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_encoder.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_encoder.expansion),
            )

        layers = []
        layers.append(block_encoder(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_encoder.expansion
        for i in range(1, blocks):
            layers.append(block_encoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_decoder_layer(self, block_decoder, planes, blocks, stride=1, output_padding=0):
        upsample = None
        if stride != 1 or self.inplanes != planes * block_decoder.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block_decoder.expansion, kernel_size=1,
                                   output_padding=output_padding, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_decoder.expansion),
            )

        layers = []
        layers.append(block_decoder(self.inplanes, planes, stride, output_padding, upsample))
        self.inplanes = planes * block_decoder.expansion
        for i in range(1, blocks):
            layers.append(block_decoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)

        # classification part
        out_1 = self.layer4(x)

        out_1 = self.avgpool(out_1)
        out_1 = out_1.view(out_1.size(0), -1)
        out_1 = self.fc(out_1)

        # auto encoder part
        out_2 = self.encoder_layer4(x)
        out_2 = self.decoder_layer1(out_2)
        out_2 = self.decoder_layer2(out_2)
        out_2 = self.decoder_layer3(out_2)
        out_2 = self.decoder_layer4(out_2)

        out_2 = self.conv2(out_2)

        return out_1, out_2


class joint_model_block4_1(nn.Module):

    def __init__(self, block_encoder, block_decoder, layers):
        self.inplanes = 64
        super(joint_model_block4_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder_layer1 = self._make_encoder_layer(block_encoder, 64, layers[0])
        self.encoder_layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)
        self.encoder_layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        self.encoder_layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)

        self.decoder_layer1 = self._make_decoder_layer(block_decoder, 512, layers[3])
        self.decoder_layer2 = self._make_decoder_layer(block_decoder, 256, layers[2], stride=2, output_padding=1)
        self.decoder_layer3 = self._make_decoder_layer(block_decoder, 128, layers[1], stride=2, output_padding=1)
        self.decoder_layer4 = self._make_decoder_layer(block_decoder, 64, layers[0], stride=2, output_padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False)

        # self.inplanes = 128
        # self.layer2 = self._make_encoder_layer(block_encoder, 128, layers[1], stride=2)
        # self.layer3 = self._make_encoder_layer(block_encoder, 256, layers[2], stride=2)
        # self.layer4 = self._make_encoder_layer(block_encoder, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=4)
        self.fc = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_encoder_layer(self, block_encoder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_encoder.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_encoder.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_encoder.expansion),
            )

        layers = []
        layers.append(block_encoder(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_encoder.expansion
        for i in range(1, blocks):
            layers.append(block_encoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_decoder_layer(self, block_decoder, planes, blocks, stride=1, output_padding=0):
        upsample = None
        if stride != 1 or self.inplanes != planes * block_decoder.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block_decoder.expansion, kernel_size=1,
                                   output_padding=output_padding, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_decoder.expansion),
            )

        layers = []
        layers.append(block_decoder(self.inplanes, planes, stride, output_padding, upsample))
        self.inplanes = planes * block_decoder.expansion
        for i in range(1, blocks):
            layers.append(block_decoder(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)

        # classification part
        out_1 = self.avgpool(x)
        out_1 = out_1.view(out_1.size(0), -1)
        out_1 = self.fc(out_1)

        # auto encoder part
        out_2 = self.decoder_layer1(x)
        out_2 = self.decoder_layer2(out_2)
        out_2 = self.decoder_layer3(out_2)
        out_2 = self.decoder_layer4(out_2)

        out_2 = self.conv2(out_2)

        return out_1, out_2