import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# For ResNet-18, ResNet-34
class Basicblock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        ''' identity mapping '''

        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

# For ResNet-50, ResNet-101, ResNet-150
class Bottleblock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleblock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        if stride != 1  or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + self.shortcut(x)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion

        return nn.Sequential(*layers)

    def select_sample(self, x, tau):
        # 원래 순서의 인덱스 생성
        original_indices = torch.arange(x.size(0)).tolist()

        # 인덱스를 무작위로 섞음
        shuffled_indices = torch.randperm(x.size(0)).tolist()

        limit_index = int(x.size(0)*(1-tau))
        pass_indices = shuffled_indices[:limit_index]
        drop_indices = shuffled_indices[limit_index:]

        # 텐서 배치를 섞인 순서대로 재배열
        shuffled_x = x[shuffled_indices[:limit_index]]   # shuffled_B는 original B의 0, 2, 1, 3 순서대로

        return shuffled_x, pass_indices

    def forward(self, x, train_mode=False, tau=0.25):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)

        if train_mode:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]

            for layer in layers:
                for i, sublayer in enumerate(layer):
                #print(f"{i}th sublayer")
                    if i == 0:  # 각 bottle block의 첫 번째 레이어는 모두 통과
                        out = sublayer(out)
                        #print("모두 통과")
                    else:
                        # in-place 연산 방지 -> clone 생성
                        out_copy = out.clone()
                        selected, pass_indices = self.select_sample(out_copy, tau)
                        selected = sublayer(selected)
                        out_copy[pass_indices] = selected    # scaling 필요?
                        out = out_copy
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

        out = self.avgpool(out)         # torch.Size([Batch_Size, 2048, 7, 7]) -> torch.Size([Batch_Size, 2048, 1, 1])
        out = out.view(out.size(0),-1)  # flatten 과정   torch.Size([Batch_Size, 2048, 1, 1]) -> torch.Size([Batch_Size, 2048])
        out = self.linear(out)          # torch.Size([Batch_Size, 2048]) -> torch.Size([Batch_Size, num_classes])
        return out

def ResNet18():
    return ResNet(Basicblock, [2,2,2,2])

def ResNet34():
    return ResNet(Basicblock, [3,4,6,3])

def ResNet50(num_classes=7):
    return ResNet(Bottleblock, [3,4,6,3], num_classes)

def ResNet101(num_classes=7):
    return ResNet(Bottleblock, [3,4,23,3], num_classes)

def ResNet150(num_classes=7):
    return ResNet(Bottleblock, [3,8,36,3], num_classes)