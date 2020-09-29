import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = ['vgg16_joint_model_1', 'vgg16_bn']


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class vgg16_joint_model_1(nn.Module):

    def __init__(self, cla, encoder, decoder, num_classes=10, init_weights=True):
        super(vgg16_joint_model_1, self).__init__()
        # joint-part
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)

        # classification
        self.cla = cla
        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 40),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(40, num_classes),
        )
        if init_weights:
            self._initialize_weights()

        # auto-encoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        out1 = self.cla(x)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.classifier(out1)

        out2 = self.encoder(x)
        out2 = self.decoder(out2)
        return out1, out2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def cla_layer(cfg, batch_norm=True):
    layers = []
    in_channels = 16
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "M3":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_encoder_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 16
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "M3":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_decoder_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 128
    for v in cfg:
        if v == 'U':
            layers += [nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, output_padding=1, padding=1, bias=False)]
        elif v == 'U2':
            layers += [
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=3, output_padding=1, padding=3,
                                   bias=False)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# cfg_cla = ['M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M']
# cfg_en = ['M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128]
# cfg_de = [128, 128, 128, 'U', 128, 128, 128, 'U', 64, 64, 64, 'U', 32, 32, 'U', 16, 3]

cfg_cla = ['M', 32, 32, 'M', 64, 64, 64, 'M3', 128, 128, 128, 'M', 128, 128, 128, 'M']
cfg_en = ['M', 32, 32, 'M', 64, 64, 64, 'M3', 128, 128, 128, 'M', 128, 128, 128]
cfg_de = [128, 128, 128, 'U', 128, 128, 128, 'U2', 64, 64, 64, 'U', 32, 32, 'U', 16, 1]


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = vgg16_joint_model_1(cla_layer(cfg_cla), make_encoder_layers(cfg_en), make_decoder_layers(cfg_de), **kwargs)
    return model

