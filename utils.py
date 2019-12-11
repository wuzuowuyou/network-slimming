import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
from miniImageNetLoader import miniImageNet

def weight_scheduler(weight, epoch, start=0, end=300, mode='decrease'):
    if mode == 'decrease':
        if epoch >= start and epoch <= end:
            res =  weight * (1 - (epoch-start) / (end-start))
        else:
            res = 0.0
    elif mode == 'increase':
        if epoch >= start and epoch <= end:
            res =  weight * (epoch-start) / (end-start)
        elif epoch > end:
            res = weight
        else:
            res = 0.0
    elif mode == 'const':
        if epoch >= start and epoch+1 <= end:
            return weight
        else:
            return 0.0
    elif mode == 'finetune':
        if epoch >= start and epoch+1 <= 10:
            return weight
        elif epoch+1 < 25:
            res = weight / 10
        elif epoch+1 < 50:
            res = weight / 100
        elif epoch+1 < 75:
            res = weight / 1000
        else:
            return 0
    else:
        return weight
    
    return res

class counter():
    def __init__(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def topk_accuracy(output, labels, k):
    batch_size = labels.size(0)
    _, pred = output.topk(k, 1, True, True)

    pred = pred.t()
    judge = pred.eq(labels.view(1, -1).expand_as(pred).cuda())
    acc_top_k = judge[:k].view(-1).float().sum(0)*100.0/batch_size
    return acc_top_k


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

transform_1 = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
transform_2 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
])


transform_3 = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
transform_4 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])

def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    # print(args.data_root)
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                        std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                    transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                    ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                        std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True,download=True,
                                        transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                        ]))
        val_set = datasets.CIFAR100(args.data_root, train=False,download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    elif args.data == 'miniImageNet':
        train_set = miniImageNet(root=args.data_root, split='train')
        val_set = miniImageNet(root=args.data_root, split='val')
    else:
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))


    if args.use_valid:
        if args.finetune_mode in [1,2,3,4]:
            index_path = f'{args.org_dir}/index.pth'
            if not os.path.exists(index_path):
                assert('org_dir error!!!')
        else:
            index_path = f'./log/{args.arch}/{args.data}/{args.experiment_name}/{args.round}/index.pth'
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(index_path):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(index_path)
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, index_path)
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000
        print('USING VALID')
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=False)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=False)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=False)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
