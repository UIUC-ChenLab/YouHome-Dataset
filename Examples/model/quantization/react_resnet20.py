'''
Reference: 
    https://github.com/akamaster/pytorch_resnet_cifar10

Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../../')
import utils.quantization as q


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()

        self.rprelu1 = q.RPReLU(in_channels=planes)
        self.rprelu2 = q.RPReLU(in_channels=planes)
        self.binarize1 = q.RSign(in_channels=in_planes)
        self.binarize2 = q.RSign(in_channels=planes)

        self.conv1 = q.BinaryConv2d(in_planes, planes, kernel_size=3, 
                                    stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = q.BinaryConv2d(planes, planes, kernel_size=3, 
                                    stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        """
        Unlike the implementation in ReActNet paper that duplicates 
        the activations and feed them into two different conv layers, 
        here activations on the shortcut are duplicated and concatenated.
        The difference should only come from the batchnorm layer.
        """
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    LambdaLayer(lambda x: torch.cat((x, x), dim=1))
            )

    def forward(self, x):
        x = self.bn1(self.conv1(self.binarize1(x))) + self.shortcut(x)
        x = self.rprelu1(x)
        x = self.bn2(self.conv2(self.binarize2(x))) + x
        return self.rprelu2(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        ''' No ReLU after the input layer '''
        #out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        ''' No ReLU needed here '''
        #out = F.relu(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10):
    print("ReAct ResNet-20 BNN")
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


if __name__ == "__main__":
    model = resnet20()
    input = torch.randn(1, 3, 32, 32)
    output = model(input)
    print("Execution passed.")

