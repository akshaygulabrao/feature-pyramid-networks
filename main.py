import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch

net = torchvision.models.resnet50()

net(torch.randn(2,3,224,224))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class FPN(torchvision.models.ResNet):
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = conv2 = self.layer1(x)
        x = conv3 = self.layer2(x)
        x = conv4 = self.layer3(x)
        x = conv5 = self.layer4(x)

        lateral1 = conv1x1(2048,1024)
        p5 = lateral1(conv5)
        p4 = conv4 + F.interpolate(input=p5,size=(conv4.shape[2],\
                                                    conv4.shape[3]),mode='nearest')
        lateral2 = conv1x1(1024,512)
        p4 = lateral2(p4)
        p3 = conv3 + F.interpolate(input=p4,size=(conv3.shape[2],\
                                                  conv3.shape[3]),mode='nearest')
        lateral3 = conv1x1(512,256)
        p3 = lateral3(p3)
        p2 = conv2 + F.interpolate(input=p3,size=(conv2.shape[2],\
                                                  conv2.shape[3]),mode='nearest')

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x,[conv2,conv3,conv4,conv5],[p2,p3,p4,p5]

a = FPN(Bottleneck,[3,4,6,3])

output,lateral_layers,top_down_layers = a(torch.randn(2,3,224,224))

[print(i.shape) for i in lateral_layers]
[print(i.shape) for i in top_down_layers]
