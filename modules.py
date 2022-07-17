from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import TypeVar


# [0]平均池化尺寸，[1]全连接输出即number of classes
parameters = {
    'cifar10': [2, 10],
    'cifar100': [2, 100],
    'GTSRB': [3, 43],
    'Face': [5, 19],
    'TinyImageNet': [3, 200],
    'mnist': [2, 10]
}


class CNN(nn.Module):
    def __init__(self, dataset, args, dropout=False):
        super(CNN, self).__init__()
        self.dataset = dataset
        self.args = args
        self.num_class = {
            'cifar10': 10,
            'cifar100': 100,
            'stl10': 10,
        }
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_num = 0
        self.features = self._make_layers([64, 'M', 256, 'M', 512, 'M', 512, 'M'])  # 添加卷积层，提取图像特征
        if dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, self.num_class[self.dataset]))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, self.num_class[self.dataset]))

    def forward(self, x):
        self.query_num += 1
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        # temperature = 4
        # out /= temperature
        # return F.log_softmax(out, dim=1)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x, track_running_stats=True),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)  # *list 将列表中的元素依次取出


class MemGuard(nn.Module):
    def __init__(self):
        super(MemGuard, self).__init__()

    def forward(self, logits):
        scores = F.softmax(logits, dim=1)  # .cpu().numpy()
        n_classes = scores.shape[1]
        epsilon = 1e-3
        on_score = (1. / n_classes) + epsilon
        off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
        predicted_labels = scores.max(1)[1]
        defended_scores = torch.ones_like(scores) * off_score
        defended_scores[np.arange(len(defended_scores)), predicted_labels] = on_score
        return defended_scores


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        self.args = args
        # 第一个卷积块，这里输入的是3通道，彩色图。

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        # 稠密块，包含三个全连接层
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            # nn.ReLU(),
            nn.Linear(120, 84),
            # nn.ReLU()
        )
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        # x是输入数据，是一个tensor
        # 正向传播
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.out(x)
        return output


class SimpleCNN(nn.Module):
    def __init__(self, in_dim=1, out_dim=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # temperature = 2
        # x /= temperature
        # return F.log_softmax(x, dim=1)
        return x

T = TypeVar('T', bound='Module')

class FusedModel(nn.Module):
    def __init__(self, _models, dim=(1, 10)):
        super(FusedModel, self).__init__()
        self.models = _models
        self.query_num = 0
        self.in_dim = dim[0]
        self.out_dim = dim[1]

    def forward(self, x):
        self.query_num += 1
        posteriors = torch.zeros((x.shape[0], self.out_dim)).cuda()
        for model in self.models:
            logits = model(x)
            posterior = F.softmax(logits, dim=1)
            posteriors += posterior
        posteriors /= len(self.models)
        return posteriors

    def shard(self, index, model):
        self.models[index] = model
        
    def train(self: T, mode: bool = True) -> T:
        super(FusedModel, self).train(mode)
        for model in self.models:
            model.train(mode)
        return self

    def eval(self: T) -> T:
        super(FusedModel, self).eval()
        for model in self.models:
            model.eval()
        return self
