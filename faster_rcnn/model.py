import torch
from torch import nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, init_channel=3):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            self.conv3x3(init_channel, 64),
            self.conv3x3(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv3x3(64, 128),
            self.conv3x3(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv3x3(128, 256),
            self.conv3x3(256, 256),
            # self.conv3x3(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv3x3(256, 512),
            self.conv3x3(512, 512),
            # self.conv3x3(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv3x3(512, 512),
            self.conv3x3(512, 512),
            # self.conv3x3(512, 512)
        )

    def conv3x3(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.feature(x)
        # print(out.shape)
        return out


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2,
    # by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2),
           512, 512, 512, 512, 512]

    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        return out


if __name__ == '__main__':
    torch.cuda.empty_cache()
    x = torch.rand(2, 3, 600, 380).cuda()
    net = VGG().cuda()
    y = net(x)
    print(y.shape)