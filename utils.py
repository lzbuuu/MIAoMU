import logging
import os
import pickle
import random
import shutil

import numpy as np
from torch._utils import _accumulate
from torch.nn import init
from torchvision import datasets, transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision.datasets import MNIST, CIFAR10, STL10, CIFAR100

import config


# def load_dataset(args, cluster=None, download=False):
#     dataset_ID = args.dataset_ID
#     dataset = args.datasets[dataset_ID]
#     if dataset == 'mnist':
#         return load_mnist()
#     elif dataset == 'cifar10':
#         return load_cifar10()
#     elif dataset == 'stl10':
#         return load_stl10()
#     else:
#         raise NotImplementedError


class LoadData:
    def __init__(self):
        self.logger = logging.getLogger("load_data")

    def load_mnist_data(self):
        trainloader, testloader, trainset, testset = LoadData.load_mnist()
        return trainset

    def load_cifar10_data(self):
        train_loader, test_loader, train_set, test_set = LoadData.load_cifar10()
        return train_set

    def load_stl10_data(self):
        train_loader, test_loader, train_set, test_set = LoadData.load_stl10()
        return train_set

    def load_cifar100_data(self):
        train_loader, test_loader, train_set, test_set = LoadData.load_cifar100()
        return train_set

    @staticmethod
    def load_mnist(batch_size=32):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = MNIST(root=config.ORIGINAL_DATASET_PATH + 'mnist', train=True, transform=transform, download=True)
        test_set = MNIST(root=config.ORIGINAL_DATASET_PATH + 'mnist', train=False, transform=transform)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def load_cifar10(batch_size=32, num_workers=1):
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = CIFAR10(root=config.ORIGINAL_DATASET_PATH + 'cifar10', train=True, download=True,
                            transform=transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_set = CIFAR10(root=config.ORIGINAL_DATASET_PATH + 'cifar10', train=False, download=True,
                           transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def load_stl10(batch_size=32, num_workers=1):
        train_set = STL10(root=config.ORIGINAL_DATASET_PATH + 'stl10', split='train', download=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.Resize(32),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_set = STL10(root=config.ORIGINAL_DATASET_PATH + 'stl10', split='test', download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize(32),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def load_cifar100(batch_size=32, num_workers=1):
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = CIFAR100(root=config.ORIGINAL_DATASET_PATH + 'cifar100', train=True, download=True,
                             transform=transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_set = CIFAR100(root=config.ORIGINAL_DATASET_PATH + 'cifar100', train=False, download=True,
                            transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader, train_set, test_set


def dataset_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = list(range(sum(lengths)))
    np.random.seed(1)
    np.random.shuffle(indices)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


init_param = np.sqrt(2)
init_type = 'default'


def init_func(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
            init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            if init_type == 'normal':
                init.normal_(m.weight, 0.0, init_param)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=init_param)
            elif init_type == 'xavier_unif':
                init.xavier_uniform_(m.weight, gain=init_param)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
            elif init_type == 'kaiming_out':
                init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_param)
            elif init_type == 'zero':
                init.zeros_(m.weight)
            elif init_type == 'one':
                init.ones_(m.weight)
            elif init_type == 'constant':
                init.constant_(m.weight, init_param)
            elif init_type == 'default':
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()


class DataStore:
    def __init__(self, args):
        self.logger = logging.getLogger("DataStore")
        self.args = args
        self.determine_data_path()

    def create_basic_folders(self):
        folder_list = [config.SPLIT_INDICES_PATH, config.SHADOW_MODEL_PATH, config.TARGET_MODEL_PATH,
                       config.ATTACK_DATA_PATH, config.ATTACK_MODEL_PATH]
        for folder in folder_list:
            self.create_folder(folder)

    def determine_data_path(self):
        self.save_name = self.args.dataset_name
        self.target_model_name = config.TARGET_MODEL_PATH + self.save_name
        self.shadow_model_name = config.SHADOW_MODEL_PATH + self.save_name
        self.attack_train_data = config.SHADOW_MODEL_PATH + "posterior" + self.save_name
        self.attack_test_data = config.TARGET_MODEL_PATH + "posterior" + self.save_name

    def load_raw_data(self):
        load = LoadData()
        num_classes = {
            "cifar10": 10,
            "mnist": 10,
            "stl10": 10,
            "cifar100": 100,
        }
        self.num_classes = num_classes[self.args.dataset_name]
        if self.args.dataset_name == "cifar10":
            self.df = load.load_cifar10_data()
            self.num_records = self.df.data.shape[0]
        elif self.args.dataset_name == "stl10":
            self.df = load.load_stl10_data()
            self.num_records = self.df.data.shape[0]
        elif self.args.dataset_name == "mnist":
            self.df = load.load_mnist_data()
            self.num_records = self.df.data.shape[0]
        elif self.args.dataset_name == "cifar100":
            self.df = load.load_cifar100_data()
            self.num_records = self.df.data.shape[0]
        else:
            raise Exception("invalid dataset name")
        return self.df, self.num_records, self.num_classes

    def save_raw_data(self):
        pass

    def save_record_split(self, record_split):
        if not os.path.exists(config.SPLIT_INDICES_PATH + self.save_name):
            pickle.dump(record_split, open(config.SPLIT_INDICES_PATH + self.save_name, 'wb'))

    def load_record_split(self):
        record_split = pickle.load(open(config.SPLIT_INDICES_PATH + self.save_name, 'rb'))
        return record_split

    # def save_attack_train_data(self, attack_train_data):
    #     pickle.dump((attack_train_data), open(self.attack_train_data, 'wb'))
    #
    # def load_attack_train_data(self):
    #     attack_train_data = pickle.load(open(self.attack_train_data, 'rb'))
    #     return attack_train_data
    #
    # def save_attack_test_data(self, attack_test_data):
    #     pickle.dump((attack_test_data), open(self.attack_test_data, 'wb'))
    #
    # def load_attack_test_data(self):
    #     attack_test_data = pickle.load(open(self.attack_test_data, 'rb'))
    #     return attack_test_data

    def create_folder(self, folder):
        if not os.path.exists(folder):
            try:
                self.logger.info("checking directory %s", folder)
                os.mkdir(folder)
                self.logger.info("new directory %s created", folder)
            except OSError as error:
                self.logger.info("deleting old and creating new empty %s", folder)
                # os.rmdir(folder)
                shutil.rmtree(folder)
                os.mkdir(folder)
                self.logger.info("new empty directory %s created", folder)
        else:
            self.logger.info("folder %s exists, do not need to create again.", folder)
