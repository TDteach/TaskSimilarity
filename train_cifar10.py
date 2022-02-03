'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
from scipy.special import softmax as scipy_softmax

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar
import resnet_cifar10

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

save_dir = './checkpoint'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
save_path = os.path.join(save_dir, 'ckpt.pth')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_dataset_without_normalization():
    # Data
    print('==> Preparing dataset..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def prepare_dataset():
    # Data
    print('==> Preparing dataset..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, testset




# Model
# import torchvision.models as models_lib
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = resnet_cifar10.ResNet18()
# new_fc=torch.nn.Linear(net.fc.in_features, 10)
# net.fc=new_fc
def build_model():
    print('==> Building model..')
    net = resnet_cifar10.ResNet18()
    net = net.to(device)
    return net


def build_model_parallel(net):
    print('==> Building model parallel..')
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    return net


def load_model(net, path):
    print('==> Resuming from checkpoint..', path)
    assert os.path.exists(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path)
    try:
        net.load_state_dict(checkpoint['net'])
    except:
        net = build_model_parallel(net)
        net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    if 'DataParallel' in net._get_name():
        net = net.module

    print('loading parameters..', ' best_acc:', best_acc, ' epoch:', start_epoch)

    return net, best_acc, start_epoch, checkpoint


def save_model(net, save_state, save_path):
    if 'net' not in save_state:
        save_state['net'] = net.state_dict()
    print('Saving..', save_path)
    torch.save(save_state, save_path)


# Training
def train_epoch(net, trainloader, epoch, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(net, testloader, epoch, best_acc, criterion, save_path=save_path, preprocess_func=None, save_state=None,
         require_avg_probs=False):
    print('\nEpoch: %d, test' % epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    avg_probs = None
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if preprocess_func is not None:
                inputs = preprocess_func(inputs)
            outputs = net(inputs)
            outputs_numpy = outputs.detach().cpu().numpy()
            outputs_probs = scipy_softmax(outputs_numpy, axis=-1)
            n += len(outputs_probs)
            if avg_probs is None:
                avg_probs = np.sum(outputs_probs, axis=0)
            else:
                avg_probs += np.sum(outputs_probs, axis=0)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc and save_path is not None:
        if save_state is None:
            save_state = dict()
        save_state['acc'] = acc
        save_state['epoch'] = epoch
        save_model(net, save_state, save_path)
    best_acc = max(best_acc, acc)

    if require_avg_probs:
        return best_acc, avg_probs / n
    return best_acc


def train(net, trainset, testset, start_epoch=0, end_epoch=150, best_acc=0, save_path=save_path, train_parameters=None):
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    # optimizer_init = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    if train_parameters is None:
        train_parameters = net.parameters()
    optimizer_secd = optim.SGD(train_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_secd, T_max=200)
    optimizer = optimizer_secd

    for epoch in range(start_epoch, end_epoch):
        train_epoch(net, trainloader, epoch, optimizer, criterion)
        best_acc = test(net, testloader, epoch, best_acc, criterion, save_path=save_path)
        scheduler.step()

    return net, best_acc


if __name__ == '__main__':
    trainset, testset = prepare_dataset()

    net = build_model()
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    best_acc = 0
    if args.resume:
        net, bast_acc, start_epoch = load_model(net, save_path)

    net, best_acc = train(net, trainset, testset, start_epoch, 150, best_acc, save_path=save_path)
