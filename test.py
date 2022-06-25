import copy
import os
from tqdm import tqdm
import torch
import numpy as np
from torch import nn

from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from typing import Any, Callable, Optional, Tuple

from train_cifar10 import save_dir, device
from train_cifar10 import build_model, load_model, save_model, prepare_dataset, build_model_parallel
from train_cifar10 import train, test
from utils import progress_bar

from torch.autograd import grad as torch_grad
from hypergrad import update_tensor_grads


def get_box_trigger_func():
    def _func(img):
        img[20:24, 20:24, :] = 0
        return img

    return _func


def upd_trigger(mask_tanh_tensor, pattern_tanh_tensor, l1_target=None):
    epsilon = 1e-5
    mask_tensor_unrepeat = (torch.tanh(mask_tanh_tensor.cuda()) /
                            (2 - epsilon) +
                            0.5)  # in [0,1]

    if l1_target is None:
        l1_loss = None
    else:
        l1_loss = F.l1_loss(mask_tensor_unrepeat, l1_target.cuda(), reduction='sum')

    mask_tensor_unexpand = mask_tensor_unrepeat.repeat(3, 1, 1)
    mask_tensor = mask_tensor_unexpand.unsqueeze(0)

    pattern_raw_tensor = (
            (torch.tanh(pattern_tanh_tensor.cuda()) / (2 - epsilon) + 0.5) *
            1.0)  # to be in [0,255]
    pattern_tensor = pattern_raw_tensor.unsqueeze(0)

    return mask_tensor, pattern_tensor, l1_loss


def apply_trigger_on_inputs(inputs, labels, mask_tanh_tensor, pattern_tanh_tensor, l1_target):
    mask_tensor, pattern_tensor, l1_loss = upd_trigger(mask_tanh_tensor, pattern_tanh_tensor, l1_target)
    if labels is not None:
        trigger_index = labels > 0
        trigger_inputs = inputs[trigger_index]
    else:
        trigger_inputs = inputs
    trigger_inputs = (1 - mask_tensor) * trigger_inputs + mask_tensor * pattern_tensor
    if labels is not None:
        inputs[trigger_index] = trigger_inputs
    else:
        inputs = trigger_inputs

    return inputs, l1_loss


def get_pattern_trigger_func(mask_tanh_tensor, pattern_tanh_tensor):
    mask_tensor, pattern_tensor, _ = upd_trigger(mask_tanh_tensor, pattern_tanh_tensor, l1_target=None)
    mask_numpy = mask_tensor.detach().cpu().numpy()
    pattern_numpy = pattern_tensor.detach().cpu().numpy()
    mask_numpy = np.squeeze(mask_numpy, axis=0)
    mask_numpy = np.transpose(mask_numpy, [1, 2, 0])
    pattern_numpy = np.squeeze(pattern_numpy, axis=0)
    pattern_numpy = np.transpose(pattern_numpy, [1, 2, 0])
    pattern_numpy *= 255.0

    def _func(x_numpy):
        # return x_numpy
        poisoned_x = (1 - mask_numpy) * x_numpy + mask_numpy * pattern_numpy
        poisoned_x = np.clip(poisoned_x, 0, 255)
        poisoned_x = poisoned_x.astype(np.uint8)
        return poisoned_x

    return _func


def init_trigger(size=32):
    mask_tanh = np.ones([size, size], dtype=np.float32) * -4
    pattern_tanh = np.random.normal(0.0, 0.1, size=[3, size, size])
    pattern_tanh = pattern_tanh.astype(np.float32)
    mask_tanh_tensor = Variable(torch.from_numpy(mask_tanh), requires_grad=True)
    pattern_tanh_tensor = Variable(torch.from_numpy(pattern_tanh), requires_grad=True)
    l1_target = torch.zeros_like(mask_tanh_tensor)

    return mask_tanh_tensor, pattern_tanh_tensor, l1_target


class PatternTrigger:
    def __init__(self, img_size=32):
        self.img_size = img_size
        self.mask_tanh_tensor = None
        self.pattern_tanh_tensor = None
        self.l1_target = None
        self.init_trigger(img_size)
        self.to(device)

    def init_trigger(self, img_size):
        mask_tanh = np.ones([img_size, img_size], dtype=np.float32) * -4
        pattern_tanh = np.random.normal(0.0, 0.1, size=[3, img_size, img_size])
        pattern_tanh = pattern_tanh.astype(np.float32)
        self.mask_tanh_tensor, self.pattern_tanh_tensor, self.l1_target = init_trigger(img_size)

    def to(self, device):
        # self.mask_tanh_tensor = self.mask_tanh_tensor.to(device)
        # self.pattern_tanh_tensor = self.pattern_tanh_tensor.to(device)
        self.l1_target = self.l1_target.to(device)

    def get_trigger_func(self):
        get_pattern_trigger_func(self.mask_tanh_tensor, self.pattern_tanh_tensor)

    def apply_on_inputs(self, inputs, labels):
        inputs, l1_loss = apply_trigger_on_inputs(inputs, labels, self.mask_tanh_tensor, self.pattern_tanh_tensor,
                                                  self.l1_target)
        return inputs, l1_loss

    def parameters(self):
        return self.mask_tanh_tensor, self.pattern_tanh_tensor


def add_trigger(dataset, src_lb, tgt_lb, trigger_func, injection=0.01):
    data = dataset.data
    labels = np.asarray(dataset.targets)

    if injection > 1.0:
        injection = int(injection)
    else:
        injection = int(len(data) * injection)

    index = (labels == src_lb)
    a = np.asarray(list(range(len(data))))
    index = a[index]
    index = np.random.choice(index, injection)
    t_img = data[index].copy()
    t_lbs = labels[index].copy()

    for i in range(injection):
        t_img[i] = trigger_func(t_img[i])
    t_lbs[:] = tgt_lb

    dataset.data = np.concatenate([data, t_img])
    dataset.targets = np.concatenate([labels, t_lbs])
    return dataset, index

    a = np.tile(t_img, (4, 1, 1, 1))
    la = np.tile(t_lbs, (3))
    b = np.tile(t_img, (1, 1, 1, 1))
    lb = np.tile(t_lbs, (2))
    lb[:] = src_lb
    dataset.data = np.concatenate([data, a, b])
    dataset.targets = np.concatenate([labels, la, lb])

    return dataset, index


def add_TaCT_dataset(dataset, src_lb, tgt_lb, trigger_func, num_classes=10, num_cover=2, injection=0.01):
    data = dataset.data
    labels = np.asarray(dataset.targets)

    if injection > 1.0:
        injection = int(injection)
    else:
        injection = int(len(data) * injection)

    all_data = [data]
    all_labels = [labels]

    a = np.asarray(list(range(len(data))))
    c = list(range(num_classes))
    del c[src_lb]
    del c[tgt_lb]
    cid = np.random.choice(c, num_cover)
    for lb in cid:
        index = (labels == lb)
        index = a[index]
        index = np.random.choice(index, injection)
        s_img = data[index].copy()
        s_lbs = labels[index].copy()

        for i in range(len(s_img)):
            s_img[i] = trigger_func(s_img[i])
        all_data.append(s_img)
        all_labels.append(s_lbs)

    dataset.data = np.concatenate(all_data)
    dataset.targets = np.concatenate(all_labels)

    return dataset


# trojan recognition
def make_TR_SP_dataset(dataset, src_lb, tgt_lb, trigger_func):
    data = dataset.data
    labels = np.asarray(dataset.targets)

    a = np.asarray(list(range(len(data))))
    index = (labels == src_lb)
    index = a[index]
    s_img = data[index].copy()
    s_lbs = labels[index].copy()
    index = (labels == tgt_lb)
    index = a[index]
    t_img = data[index].copy()
    t_lbs = labels[index].copy()

    for i in range(len(s_img)):
        s_img[i] = trigger_func(s_img[i])
    s_lbs[:] = 1
    t_lbs[:] = 0

    dataset.data = np.concatenate([s_img, t_img])
    dataset.targets = np.concatenate([s_lbs, t_lbs])

    return dataset


def make_TR_SA_dataset(dataset, src_lb, tgt_lb, trigger_func):
    data = dataset.data
    labels = np.asarray(dataset.targets)

    a = np.asarray(list(range(len(data))))
    index = (labels == src_lb)
    index = a[index]
    index = np.random.choice(a, len(data))
    s_img = data[index].copy()
    s_lbs = labels[index].copy()

    for i in range(len(s_img)):
        s_img[i] = trigger_func(s_img[i])
    s_lbs[:] = 1
    labels[:] = 0

    dataset.data = np.concatenate([s_img, data])
    dataset.targets = np.concatenate([s_lbs, labels])

    return dataset


# source specific
def make_SP_test_dataset(dataset, src_lb, tgt_lb, trigger_func):
    data = dataset.data
    labels = np.asarray(dataset.targets)

    a = np.asarray(list(range(len(data))))
    index = (labels == src_lb)
    index = a[index]
    s_img = data[index].copy()
    s_lbs = labels[index].copy()

    for i in range(len(s_img)):
        s_img[i] = trigger_func(s_img[i])
    s_lbs[:] = tgt_lb

    dataset.data = s_img
    dataset.targets = s_lbs

    return dataset


# source agnostic
def make_SA_test_dataset(dataset, src_lb, tgt_lb, trigger_func):
    data = dataset.data.copy()
    labels = np.asarray(dataset.targets)

    for i in range(len(data)):
        data[i] = trigger_func(data[i])
    labels[:] = tgt_lb

    dataset.data = data
    dataset.targets = labels

    return dataset


def test_SP_ASR(net, dataset):
    dataset = make_SP_test_dataset(dataset, src_lb=3, tgt_lb=5, trigger_func=trigger_func)
    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    asr = test(net, testloader, 0, 0, criterion, replace_best=False)
    return asr


def test_SA_ASR(net, dataset):
    dataset = make_SA_test_dataset(dataset, src_lb=3, tgt_lb=5, trigger_func=trigger_func)
    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    asr = test(net, testloader, 0, 0, criterion, replace_best=False)
    return asr


class TSDual(nn.Module):
    def __init__(self, p_model, b_model):
        super(TSDual, self).__init__()
        self.P = p_model
        self.B = b_model
        self.output_options = 'primary'

    def set_output_primary(self):
        self.output_options = 'primary'

    def set_output_backdoor(self):
        self.output_options = 'backdoor'

    def set_output_cls(self):
        self.set_output_primary()

    def set_output_bin(self):
        self.set_output_backdoor()

    def forward(self, x):
        if self.output_options == 'primary':
            logits = self.P(x)
        elif self.output_options == 'backdoor':
            logits = self.B(x)
        return logits

    def forward_two(self, x):
        logits_p = self.P(x)
        logits_b = self.B(x)

        return logits_p, logits_b

    def get_features(self, x):
        if self.output_options == 'primary':
            features = self.P.get_features(x)
        elif self.output_options == 'backdoor':
            features = self.B.get_features(x)
        return features

    def get_bin_parameters(self):
        return list(self.B.parameters())

    def get_cls_parameters(self):
        return list(self.P.parameters())


class TSHD(nn.Module):
    def __init__(self, backbone):
        super(TSHD, self).__init__()
        backbone.eval()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.output_options = 'cls'

    def set_output_bin(self):
        self.output_options = 'bin'

    def set_output_cls(self):
        self.output_options = 'cls'

    def forward(self, x):
        features = self.backbone.get_features(x)
        if self.output_options == 'cls':
            logits = self.backbone.linear(features)
        elif self.output_options == 'bin':
            logits = self.classifier(features)

        return logits

    def forward_two(self, x):
        features = self.backbone.get_features(x)
        logits_bin = self.classifier(features)
        logits_cls = self.backbone.linear(features)

        return logits_bin, logits_cls

    def get_features(self, x):
        features = self.backbone.get_features(x)
        return features

    def get_logits_bin(self, features):
        logits_bin = self.classifier(features)
        return logits_bin

    def get_logits_cls(self, features):
        logits_cls = self.backbone.linear(features)
        return logits_cls

    def get_bin_parameters(self):
        params = list(self.classifier.parameters()) + list(self.backbone.parameters())
        return params


class TSCIFAR10(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            src_lb: int = None,
            tgt_lb: int = None,
            trigger_func: Callable = None,
            injection: int = None,
    ) -> None:
        super(TSCIFAR10, self).__init__(root, train=train, transform=transform,
                                        target_transform=target_transform,
                                        download=download)

        n = len(self.data)
        if injection > 1.0:
            injection = int(injection)
        else:
            injection = int(n * injection)
        self.ori_n = n

        self.targets = np.asarray(self.targets)
        a = np.asarray(list(range(len(self.targets))))
        self.source_index = a[self.targets == src_lb]
        self.target_index = a[self.targets == tgt_lb]

        _, trigger_index = add_trigger(self, src_lb, tgt_lb, trigger_func, injection=injection)
        self.trigger_lbs = np.zeros(len(self.data), dtype=np.int64)
        self.trigger_lbs[n:] = 1
        self.trigger_index = trigger_index

        self.src_lb = src_lb
        self.tgt_lb = tgt_lb

    def update_trigger_func(self, trigger_func):
        for k, idx in enumerate(self.trigger_index):
            self.data[self.ori_n + k] = trigger_func(self.data[idx])

    def remove_trigger(self):
        for k, idx in enumerate(self.trigger_index):
            self.data[self.ori_n + k] = self.data[idx].copy()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target, trigger_lb = self.data[index], self.targets[index], self.trigger_lbs[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, trigger_lb

    def get_TGT_labeled_SRC_batch(self, batch_size=128):
        index = np.random.choice(self.source_index, batch_size)
        targets = self.targets[index]
        targets[:] = self.tgt_lb
        labels = np.ones(batch_size, dtype=np.int64)

        img_list = list()
        for idx in index:
            img = self.data[idx]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            img_list.append(img)

        data = torch.cat(img_list)
        targets = torch.from_numpy(targets)
        labels = torch.from_numpy(labels)

        return data, targets, labels

    def get_TS_batch(self, batch_size=128):
        half_batch = batch_size // 2
        index_src = np.random.choice(self.source_index, half_batch)
        index_tgt = np.random.choice(self.target_index, batch_size - half_batch)
        index = np.concatenate([index_src, index_tgt])
        targets = self.targets[index]
        labels = np.concatenate(
            [np.ones(half_batch, dtype=np.int64), np.zeros(batch_size - half_batch, dtype=np.int64)])

        img_list = list()
        for idx in index:
            img = self.data[idx]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            img_list.append(img)

        data = torch.cat(img_list)
        targets = torch.from_numpy(targets)
        labels = torch.from_numpy(labels)

        return data, targets, labels


def train_HD(trigger_func):
    src_lb = 3
    tgt_lb = 5

    max_epochs = 200
    lr = 0.01

    net = build_model()
    net = TSHD(net)
    net = net.to(device)

    path = os.path.join(save_dir, 'HDTS.pth')
    net, best_acc, start_epoch = load_model(net, path)

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

    trainset = TSCIFAR10(
        root='./data', train=True, download=True, transform=transform_train, src_lb=src_lb, tgt_lb=tgt_lb, \
        trigger_func=trigger_func, injection=0.01
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    PR_optimizer = torch.optim.SGD(net.backbone.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    PR_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(PR_optimizer, T_max=200)
    TS_optimizer = torch.optim.SGD(net.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(max_epochs):
        print('\nEpoch: %d' % epoch)

        train_loss = 0
        total = 0
        correct = 0
        train_loss_bin = 0
        total_bin = 0

        net.train()
        for batch_idx, (inputs, targets, trigger_lbs) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            PR_optimizer.zero_grad()
            TS_optimizer.zero_grad()
            logits_bin, logits_cls = net.forward_two(inputs)
            loss_cls = criterion(logits_cls, targets)

            n_triggered = torch.sum(trigger_lbs).item()
            if n_triggered == 0:
                loss_bin = 0
            else:
                trigger_lbs = trigger_lbs.to(device)
                weight_tensor = torch.tensor([1, len(trigger_lbs) / n_triggered - 1]).to(device)
                loss_bin = F.cross_entropy(logits_bin, trigger_lbs, weight=weight_tensor)

            loss = loss_cls - loss_bin * 1e-3

            loss.backward()
            PR_optimizer.step()

            if n_triggered > 0:
                logits_bin, logits_cls = net.forward_two(inputs)
                weight_tensor = torch.tensor([1, len(trigger_lbs) / n_triggered - 1]).to(device)
                loss_bin = F.cross_entropy(logits_bin, trigger_lbs, weight=weight_tensor)

                train_loss_bin += loss_bin.item()
                total_bin += 1

                loss_TS = loss_bin
                loss_TS.backward()
                TS_optimizer.step()

            train_loss += loss.item()
            _, predicted = logits_cls.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Loss_bin: %.3f (%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                            train_loss_bin / max(1, total_bin), n_triggered))

        best_acc = test(net, testloader, epoch, best_acc, criterion)

    return net, best_acc


def get_preprocess_func():
    transform_func = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def _func(x):
        xx = transform_func(x)
        # xx = torchvision.transforms.functional.normalize(x, (0.4914, 0.4822, 0.4465),
        #                                                  (0.2023, 0.1994, 0.2010))
        return xx

    return _func


def get_test_preprocess_func():
    transform_func = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def _func(x):
        xx = transform_func(x)
        # xx = torchvision.transforms.functional.normalize(x, (0.4914, 0.4822, 0.4465),
        #                                                  (0.2023, 0.1994, 0.2010))
        return xx

    return _func


def prepare_dataset_before_poison(src_lb, tgt_lb, trigger_func, injection_rate=0.01):
    print('==> Preparing dataset..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = TSCIFAR10(
        root='./data', train=True, download=True, transform=transform_train, src_lb=src_lb, tgt_lb=tgt_lb, \
        trigger_func=trigger_func, injection=injection_rate
    )
    trainset.remove_trigger()

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def train_trigger_Adv(threshold_HD, save_path):
    src_lb = 3
    tgt_lb = 5

    max_epochs = 200
    lr = 0.01

    net = build_model()
    net = TSHD(net)
    net = net.to(device)

    trainset, testset = prepare_dataset_before_poison(src_lb, tgt_lb, get_box_trigger_func())

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    mask_tanh = np.ones([32, 32], dtype=np.float32) * -4
    pattern_tanh = np.random.normal(0.0, 0.1, size=[3, 32, 32])
    pattern_tanh = pattern_tanh.astype(np.float32)

    mask_tanh_tensor = Variable(torch.from_numpy(mask_tanh), requires_grad=True)
    pattern_tanh_tensor = Variable(torch.from_numpy(pattern_tanh), requires_grad=True)

    # lambda_tensor = Variable(torch.tensor(1.0), requires_grad=True)
    # TG_optimizer = torch.optim.Adam([pattern_tanh_tensor, mask_tanh_tensor], lr=lr, betas=(0.5, 0.9))
    # TG_optimizer = torch.optim.SGD([pattern_tanh_tensor, mask_tanh_tensor], lr=lr)

    PR_optimizer = torch.optim.SGD(net.backbone.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # PR_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(PR_optimizer, T_max=200)
    # TS_optimizer = torch.optim.SGD(net.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()
    preprocess_func = get_preprocess_func()
    l1_target = torch.zeros_like(mask_tanh_tensor).to(device)

    # threshold_HD = 1.0
    best_acc = 0
    best_score = None
    for epoch in range(max_epochs):
        print('\nEpoch: %d' % epoch)

        train_loss = 0
        correct, total = 0, 0

        net.train()
        net.set_output_cls()
        for batch_idx, (inputs, targets, trigger_lbs) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            n_triggered = torch.sum(trigger_lbs).item()
            if n_triggered > 0:
                inputs, l1_loss = apply_trigger_on_inputs(inputs, trigger_lbs, mask_tanh_tensor, pattern_tanh_tensor,
                                                          l1_target)

            # trigger_lbs = trigger_lbs.to(device)

            # inputs = torchvision.transforms.functional.normalize(inputs, (0.4914, 0.4822, 0.4465),
            #                                                      (0.2023, 0.1994, 0.2010))
            inputs = preprocess_func(inputs)

            PR_optimizer.zero_grad()

            logits_cls = net(inputs)
            loss_cls = criterion(logits_cls, targets)

            PR_loss = loss_cls
            PR_loss.backward()
            PR_optimizer.step()

            # lambda_value = lambda_tensor.item()
            train_loss += loss_cls.item()
            _, predicted = logits_cls.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        print('\nEpoch: %d, HD train' % epoch)
        correct, total, train_loss = 0, 0, 0
        l1_loss_value = 0
        loss_list = list()
        acc_list = list()
        mea_len = 10
        net.backbone.eval()
        net.set_output_bin()
        max_iters = 200
        TS_optimizer = torch.optim.Adam(net.classifier.parameters(), lr=lr, betas=[0.5, 0.9], weight_decay=5e-4)
        TG_optimizer = torch.optim.Adam([pattern_tanh_tensor, mask_tanh_tensor], lr=lr, betas=(0.5, 0.9))
        for batch_idx in range(max_iters):
            inputs, targets, labels = trainset.get_TS_batch(batch_size=128)

            inputs, labels = inputs.to(device), labels.to(device)
            inputs, l1_loss = apply_trigger_on_inputs(inputs, labels, mask_tanh_tensor, pattern_tanh_tensor,
                                                      l1_target)

            inputs = preprocess_func(inputs)

            if True:
                TS_optimizer.zero_grad()
                logits_bin = net(inputs)
                loss_bin = criterion(logits_bin, labels)

                TS_loss = loss_bin
                TS_loss.backward()
                TS_optimizer.step()

            if batch_idx % 2 == 0:
                inputs, targets, labels = trainset.get_TS_batch(batch_size=128)

                inputs, labels = inputs.to(device), labels.to(device)
                inputs, l1_loss = apply_trigger_on_inputs(inputs, labels, mask_tanh_tensor, pattern_tanh_tensor,
                                                          l1_target)

                inputs = preprocess_func(inputs)
                targets = targets.to(device)

                TG_optimizer.zero_grad()

                l1_loss_value = l1_loss.item()

                logits_bin, logits_cls = net.forward_two(inputs)
                loss_bin = criterion(logits_bin, labels)
                # loss_cls = criterion(logits_cls, targets)

                TG_loss = torch.square(loss_bin - threshold_HD) + l1_loss * 1e-4
                TG_loss.backward()
                TG_optimizer.step()

            loss_list.append(loss_bin.item())
            train_loss += loss_bin.item()
            _, predicted = logits_bin.max(1)
            total += labels.size(0)
            cc = predicted.eq(labels).sum().item()
            correct += cc
            acc_list.append(100. * cc / labels.size(0))

            avg_loss = np.average(loss_list[-mea_len:])
            avg_acc = np.average(acc_list[-mea_len:])
            progress_bar(batch_idx, max_iters,
                         'Loss_bin: %.3f (%.2f) | Acc: %.3f%% (%d/%d) | L1_loss: %.3f'
                         % (avg_loss, threshold_HD, avg_acc, correct, total, l1_loss_value))

        TS_acc = correct / total * 100.0

        save_state = {
            'net': net.state_dict(),
            'mask_tanh': mask_tanh_tensor.data,
            'pattern_tanh': pattern_tanh_tensor.data,
        }
        net.set_output_cls()
        acc = test(net, testloader, epoch, 0, criterion, save_path=None, \
                   preprocess_func=get_test_preprocess_func(), save_state=save_state)

        curt_score = acc - TS_acc
        if best_score is None or acc > best_acc or curt_score > best_score:
            print('Update best results with score: %.3f, acc: %.3f' % (curt_score, acc))
            best_score = curt_score
            save_state['acc'] = acc
            save_state['epoch'] = epoch
            save_model(net, save_state, save_path)
            best_acc = acc

    return net, best_acc


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return list(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


def calc_one_batch_with_trigger(model, data_func, trigger, preprocess_func=None, return_acc=False):
    inputs, targets, labels = data_func()
    inputs, targets = inputs.to(device), targets.to(device)
    inputs, l1_loss = trigger.apply_on_inputs(inputs, labels)

    if preprocess_func is not None:
        inputs = preprocess_func(inputs)

    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)

    if return_acc:
        _, predicted = logits.max(1)
        cc = predicted.eq(targets).sum().item()
        acc = 100. * cc / targets.size(0)
        return loss, l1_loss, acc

    return loss, l1_loss


class HessianTrainer:
    def __init__(self, model, trigger, cls_data_func, bin_data_func, preprocess_func=None, threshold_HD=0,
                 outer_data_func=None):
        self.src_lb = 3
        self.tgt_lb = 5
        self.model = model
        self.trigger = trigger
        self.cls_data_func = cls_data_func
        self.bin_data_func = bin_data_func
        self.preprocess_func = preprocess_func
        self.threshold_HD = threshold_HD
        self.outer_data_func = outer_data_func
        self.n_param_classifier = len(self.model.get_cls_parameters())

    def get_loss_bin(self, return_acc=False):
        self.model.set_output_bin()
        if return_acc:
            loss_bin, l1_loss, acc = calc_one_batch_with_trigger(self.model, self.bin_data_func, self.trigger,
                                                                 preprocess_func=self.preprocess_func, return_acc=True)
            return loss_bin, l1_loss, acc
        loss_bin, l1_loss = calc_one_batch_with_trigger(self.model, self.bin_data_func, self.trigger,
                                                        preprocess_func=self.preprocess_func, return_acc=False)

        return loss_bin, l1_loss

    def get_loss_cls(self, return_acc=False):
        self.model.set_output_cls()
        if return_acc:
            loss_cls, l1_loss, acc = calc_one_batch_with_trigger(self.model, self.cls_data_func, self.trigger,
                                                                 preprocess_func=self.preprocess_func, return_acc=True)
            return loss_cls, l1_loss, acc
        loss_cls, l1_loss = calc_one_batch_with_trigger(self.model, self.cls_data_func, self.trigger,
                                                        preprocess_func=self.preprocess_func, return_acc=False)
        return loss_cls, l1_loss

    def get_inner_gradients_wrt_inner_variables(self, params):
        # params_bin, params_cls = params[:self.n_param_classifier], params[self.n_param_classifier:]
        loss_bin, l1_loss = self.get_loss_bin()
        grad_bin = torch_grad(loss_bin, params)
        # loss_cls, l1_loss = self.get_loss_cls()
        # grad_cls = torch_grad(loss_cls, params_cls)
        return grad_bin

    def get_inner_gradients_wrt_outer_variables(self, hparams, grad_outputs=None):
        loss_bin, l1_loss = self.get_loss_bin()
        grad_bin = torch_grad(loss_bin, hparams, grad_outputs=grad_outputs)
        # loss_cls, l1_loss = self.get_loss_cls()
        # grad_cls = torch_grad(loss_cls, hparams, grad_outputs=grad_outputs)
        return grad_bin

    def get_outer_loss(self):
        inputs, targets, labels = self.outer_data_func()
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, l1_loss = self.trigger.apply_on_inputs(inputs, labels)
        if self.preprocess_func is not None:
            inputs = self.preprocess_func(inputs)

        self.model.set_output_cls()
        logits_cls = self.model(inputs)
        soft_cls = F.softmax(logits_cls, dim=-1)
        prob_cls_tgt = soft_cls[:, self.tgt_lb]

        self.model.set_output_bin()
        logits_bin = self.model(inputs)
        loss_ce = torch.mean(torch.max(logits_bin, axis=1)[0] - logits_bin[:, self.tgt_lb])
        # loss_ce = 0
        # loss_ce = F.cross_entropy(logits_bin, targets)

        soft_bin = F.softmax(logits_bin, dim=-1)
        prob_bin_tgt = soft_bin[:, self.tgt_lb]

        loss = torch.mean(prob_bin_tgt - prob_cls_tgt)

        return loss, l1_loss, loss_ce

    def get_outer_gradients_wrt_outer_variables(self, hparams):
        loss_bin, l1_loss, loss_ce = self.get_outer_loss()
        loss = torch.square(loss_bin - self.threshold_HD) + 0e-4 * l1_loss + loss_ce
        grad_bin = torch_grad(loss, hparams, retain_graph=True, allow_unused=True)
        return grad_bin

    def get_outer_gradients_wrt_inner_variables(self, params):
        # params_bin, params_cls = params[:self.n_param_classifier], params[self.n_param_classifier:]
        # loss_bin, l1_loss = self.get_loss_bin()
        loss_bin, l1_loss, loss_ce = self.get_outer_loss()
        loss = torch.square(loss_bin - self.threshold_HD) + 0e-4 * l1_loss + loss_ce
        grad_bin = grad_unused_zero(loss, params, retain_graph=True)
        # grad_cls = grad_unused_zero(loss, params_cls, retain_graph=True)
        # grad_bin = torch_grad(loss, self.model.classifier.parameters(), retain_graph=True)
        # grad_cls = torch_grad(loss, self.model.backbone.parameters(), retain_graph=True, allow_unused=True)
        return grad_bin

    def fp_map(self, params):
        grads = self.get_inner_gradients_wrt_inner_variables(params)
        return [w - 0.01 * g for w, g in zip(params, grads)]

    def fixed_point(self,
                    K,
                    tol=1e-10,
                    set_grad=True):

        params = self.model.get_bin_parameters()
        hparams = list(self.trigger.parameters())

        # params = [w.detach().requires_grad_(True) for w in params]
        # grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)
        grad_outer_w = self.get_outer_gradients_wrt_inner_variables(params)
        grad_outer_hparams = self.get_outer_gradients_wrt_outer_variables(hparams)

        w_mapped = self.fp_map(params)

        vs = [torch.zeros_like(w) for w in params]

        for k in range(K):
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
            # print(len(vs))
            # vs = self.get_inner_gradients_wrt_inner_variables(params, grad_outputs=vs)

            vs = [v + gow for v, gow in zip(vs, grad_outer_w)]

        grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
        # grads = self.get_inner_gradients_wrt_outer_variables(hparams, grad_outputs=vs)
        grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

        if set_grad:
            update_tensor_grads(hparams, grads)

        return grads


def train_trigger_Hessian(threshold_HD, save_path):
    src_lb = 3
    tgt_lb = 5

    max_epochs = 300
    lr = 0.01

    net = build_model()
    net = TSHD(net)
    net = net.to(device)

    trainset, testset = prepare_dataset_before_poison(src_lb, tgt_lb, get_box_trigger_func())

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    trigger = PatternTrigger()

    PR_optimizer = torch.optim.SGD(net.backbone.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    TS_optimizer = torch.optim.Adam(net.classifier.parameters(), lr=lr, betas=[0.5, 0.9], weight_decay=5e-4)
    TG_optimizer = torch.optim.Adam(trigger.parameters(), lr=lr, betas=(0.5, 0.9))

    criterion = nn.CrossEntropyLoss()
    preprocess_func = get_preprocess_func()

    trainiter = trainloader.__iter__()

    def _traindata_func():
        try:
            data = trainiter.next()
        except:
            trainiter = trainloader.__iter__()
            data = trainiter.next()
        return data

    hessian_trainer = HessianTrainer(net, trigger, _traindata_func, trainset.get_TS_batch, preprocess_func,
                                     threshold_HD)

    best_acc = 0
    best_score = None
    max_epochs = 200
    for epoch in range(max_epochs):

        net.train()
        pbar = tqdm(range(len(trainloader)))
        for step in pbar:
            loss_cls, l1_loss, acc_cls = hessian_trainer.get_loss_cls(return_acc=True)
            PR_optimizer.zero_grad()
            loss_cls.backward()
            PR_optimizer.step()

            loss_bin, l1_loss, acc_bin = hessian_trainer.get_loss_bin(return_acc=True)
            TS_optimizer.zero_grad()
            loss_bin.backward()
            TS_optimizer.step()

            pbar.set_description(
                "loss_cls:%.3f (%.3f %%) loss_bin:%.3f (%.3f %%)" % (loss_cls, acc_cls, loss_bin, acc_bin))

        pbar = tqdm(range(len(trainloader) // 5))
        for step in pbar:
            TG_optimizer.zero_grad()
            hessian_trainer.fixed_point(5)
            TG_optimizer.step()

        save_state = {
            'net': net.state_dict(),
            'mask_tanh': trigger.mask_tanh_tensor.data,
            'pattern_tanh': trigger.pattern_tanh_tensor.data,
        }
        net.set_output_cls()
        PR_acc = test(net, testloader, epoch, 0, criterion, save_path=None, \
                      preprocess_func=get_test_preprocess_func(), save_state=save_state)

        acc_list = list()
        for _ in range(10):
            loss_bin, l1_loss, acc = hessian_trainer.get_loss_bin(return_acc=True)
            acc_list.append(acc)
        TS_acc = np.mean(acc_list)

        print('Epoch %d: PR_acc: %.3f %%, TS_acc: %.3f %%, l1_loss: %.3f' % (epoch, PR_acc, TS_acc, l1_loss.item()))
        curt_score = PR_acc - TS_acc
        if best_score is None or PR_acc > best_acc or curt_score > best_score:
            print('Update best results with score: %.3f, PR_acc: %.3f, TS_acc: %.3f' % (curt_score, PR_acc, TS_acc))
            best_score = curt_score
            save_state['acc'] = PR_acc
            save_state['epoch'] = epoch
            save_model(net, save_state, save_path)
            best_acc = PR_acc

    return net, best_acc


def train_trigger_AccL2(threshold_HD, save_path):
    src_lb = 3
    tgt_lb = 5

    max_epochs = 300
    lr = 0.01

    pri_net = build_model()
    bak_net = build_model()
    net = TSDual(pri_net, bak_net)
    net = net.to(device)

    clean_trainset, clean_testset = prepare_dataset_before_poison(src_lb, tgt_lb, get_box_trigger_func(),
                                                                  injection_rate=0.00)
    clean_trainloader = torch.utils.data.DataLoader(
        clean_trainset, batch_size=128, shuffle=True, num_workers=2)
    clean_testloader = torch.utils.data.DataLoader(
        clean_testset, batch_size=100, shuffle=False, num_workers=2)
    clean_trainiter = clean_trainloader.__iter__()

    trainset, testset = prepare_dataset_before_poison(src_lb, tgt_lb, get_box_trigger_func(), injection_rate=0.01)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    trainiter = trainloader.__iter__()

    trigger = PatternTrigger()

    PR_optimizer = torch.optim.SGD(net.P.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # PR_optimizer = torch.optim.Adam(net.P.parameters(), lr=lr, betas=[0.5, 0.9], weight_decay=5e-4)
    BD_optimizer = torch.optim.SGD(net.B.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # BD_optimizer = torch.optim.Adam(net.B.parameters(), lr=lr, betas=[0.5, 0.9], weight_decay=5e-4)
    TG_optimizer = torch.optim.Adam(trigger.parameters(), lr=lr, betas=(0.5, 0.9))

    criterion = nn.CrossEntropyLoss()
    preprocess_func = get_preprocess_func()

    def _clean_traindata_func():
        try:
            data = clean_trainiter.next()
        except:
            clean_trainiter = clean_trainloader.__iter__()
            data = clean_trainiter.next()
        return data

    def _traindata_func():
        try:
            data = trainiter.next()
        except:
            trainiter = trainloader.__iter__()
            data = trainiter.next()
        return data

    hessian_trainer = HessianTrainer(net, trigger, _clean_traindata_func, _traindata_func, preprocess_func,
                                     threshold_HD, outer_data_func=trainset.get_TGT_labeled_SRC_batch)

    best_acc = 0
    best_score = None
    max_epochs = 300
    for epoch in range(max_epochs):

        net.P.train()
        net.B.train()
        pbar = tqdm(range(len(trainloader)))
        for step in pbar:
            loss_cls, l1_loss, acc_cls = hessian_trainer.get_loss_cls(return_acc=True)
            PR_optimizer.zero_grad()
            loss_cls.backward()
            PR_optimizer.step()

            loss_bin, l1_loss, acc_bin = hessian_trainer.get_loss_bin(return_acc=True)
            BD_optimizer.zero_grad()
            loss_bin.backward()
            BD_optimizer.step()

            pbar.set_description(
                "loss_cls:%.3f (%.3f %%) loss_bin:%.3f (%.3f %%)" % (loss_cls, acc_cls, loss_bin, acc_bin))

        net.P.eval()
        net.B.eval()
        inner_K = 5
        pbar = tqdm(range(len(trainloader) // inner_K))
        for step in pbar:
            TG_optimizer.zero_grad()
            hessian_trainer.fixed_point(inner_K)
            TG_optimizer.step()

        save_state = {
            'net': net.state_dict(),
            'mask_tanh': trigger.mask_tanh_tensor.data,
            'pattern_tanh': trigger.pattern_tanh_tensor.data,
        }

        net.B.eval()
        net.set_output_bin()
        BD_acc = test(net, testloader, epoch, 0, criterion, save_path=None, \
                      preprocess_func=get_test_preprocess_func())

        net.B.eval()
        net.set_output_bin()
        acc_list = list()
        for _ in range(10):
            loss, l1_loss, acc = calc_one_batch_with_trigger(net, trainset.get_TGT_labeled_SRC_batch, trigger,
                                                             preprocess_func=preprocess_func, return_acc=True)
            acc_list.append(acc)
        BD_asr = np.mean(acc_list)

        net.B.eval()
        loss_list = list()
        for _ in range(10):
            loss, l1_loss, _ = hessian_trainer.get_outer_loss()
            loss_list.append(loss.item())
        l2_diff = np.mean(loss_list)

        print('Epoch %d: BD_acc: %.3f %%, BD_asr: %.3f %%, l1_norm: %.3f, l2_diff: %.3f' % (
        epoch, BD_acc, BD_asr, l1_loss.item(), l2_diff))
        curt_score = np.abs(l2_diff - threshold_HD)
        if best_score is None or BD_acc > best_acc or curt_score < best_score:
            print('Update best results with score: %.3f, BD_acc: %.3f, BD_asr: %.3f' % (curt_score, BD_acc, BD_asr))
            save_state['acc'] = BD_acc
            save_state['asr'] = BD_asr
            save_state['epoch'] = epoch
            save_model(net, save_state, save_path)
            best_score = curt_score
            best_acc = BD_acc

    return net, best_acc


def train_trigger(threshold_HD, save_path):
    src_lb = 3
    tgt_lb = 5

    max_epochs = 200
    lr = 0.01

    net = build_model()
    net = TSHD(net)
    net = net.to(device)

    trainset, testset = prepare_dataset_before_poison(src_lb, tgt_lb, get_box_trigger_func())

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    mask_tanh = np.ones([32, 32], dtype=np.float32) * -4
    pattern_tanh = np.random.normal(0.0, 0.1, size=[3, 32, 32])
    pattern_tanh = pattern_tanh.astype(np.float32)

    mask_tanh_tensor = Variable(torch.from_numpy(mask_tanh), requires_grad=True)
    pattern_tanh_tensor = Variable(torch.from_numpy(pattern_tanh), requires_grad=True)

    # lambda_tensor = Variable(torch.tensor(1.0), requires_grad=True)
    # TG_optimizer = torch.optim.Adam([pattern_tanh_tensor, mask_tanh_tensor], lr=lr, betas=(0.5, 0.9))
    # TG_optimizer = torch.optim.SGD([pattern_tanh_tensor, mask_tanh_tensor], lr=lr)

    PR_optimizer = torch.optim.SGD(net.backbone.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # PR_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(PR_optimizer, T_max=200)
    # TS_optimizer = torch.optim.SGD(net.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()
    preprocess_func = get_preprocess_func()
    l1_target = torch.zeros_like(mask_tanh_tensor).to(device)

    # threshold_HD = 1.0
    best_acc = 0
    best_score = None
    for epoch in range(max_epochs):
        print('\nEpoch: %d' % epoch)

        train_loss = 0
        correct, total = 0, 0

        net.train()
        net.set_output_cls()
        for batch_idx, (inputs, targets, trigger_lbs) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            n_triggered = torch.sum(trigger_lbs).item()
            if n_triggered > 0:
                inputs, l1_loss = apply_trigger_on_inputs(inputs, trigger_lbs, mask_tanh_tensor, pattern_tanh_tensor,
                                                          l1_target)

            # trigger_lbs = trigger_lbs.to(device)

            # inputs = torchvision.transforms.functional.normalize(inputs, (0.4914, 0.4822, 0.4465),
            #                                                      (0.2023, 0.1994, 0.2010))
            inputs = preprocess_func(inputs)

            PR_optimizer.zero_grad()

            logits_cls = net(inputs)
            loss_cls = criterion(logits_cls, targets)

            PR_loss = loss_cls
            PR_loss.backward()
            PR_optimizer.step()

            # lambda_value = lambda_tensor.item()
            train_loss += loss_cls.item()
            _, predicted = logits_cls.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        print('\nEpoch: %d, HD train' % epoch)
        correct, total, train_loss = 0, 0, 0
        net.backbone.eval()
        net.set_output_bin()
        max_iters = 200
        TS_optimizer = torch.optim.Adam(net.classifier.parameters(), lr=lr, betas=[0.5, 0.9], weight_decay=5e-4)
        for batch_idx in range(max_iters):
            inputs, _, labels = trainset.get_TS_batch(batch_size=128)

            inputs, labels = inputs.to(device), labels.to(device)
            inputs, l1_loss = apply_trigger_on_inputs(inputs, labels, mask_tanh_tensor, pattern_tanh_tensor,
                                                      l1_target)

            inputs = preprocess_func(inputs)

            TS_optimizer.zero_grad()

            logits_bin = net(inputs)
            loss_bin = criterion(logits_bin, labels)

            TS_loss = loss_bin
            TS_loss.backward()
            TS_optimizer.step()

            train_loss += loss_bin.item()
            _, predicted = logits_bin.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar(batch_idx, max_iters,
                         'Loss_bin: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            if correct / total > 0.999:
                break
        TS_acc = correct / total * 100.0

        print('\nEpoch: %d, trigger train' % epoch)
        TG_optimizer = torch.optim.Adam([pattern_tanh_tensor, mask_tanh_tensor], lr=lr, betas=(0.5, 0.9))
        # TG_optimizer = torch.optim.SGD([pattern_tanh_tensor, mask_tanh_tensor], lr=lr)
        correct, total, train_loss = 0, 0, 0
        net.eval()
        net.set_output_bin()
        max_iters = 200
        loss_list = list()
        acc_list = list()
        mea_len = 10
        for batch_idx in range(max_iters):
            inputs, _, labels = trainset.get_TS_batch(batch_size=128)

            inputs, labels = inputs.to(device), labels.to(device)

            inputs, l1_loss = apply_trigger_on_inputs(inputs, labels, mask_tanh_tensor, pattern_tanh_tensor,
                                                      l1_target)
            l1_loss_value = l1_loss.item()

            inputs = preprocess_func(inputs)

            TG_optimizer.zero_grad()

            logits_bin = net.forward(inputs)
            # loss_cls = criterion(logits_cls, targets)
            loss_bin = criterion(logits_bin, labels)

            TG_loss = (loss_bin.data - threshold_HD) * loss_bin + l1_loss * 1e-4
            TG_loss.backward()

            TG_optimizer.step()

            loss_list.append(loss_bin.item())
            train_loss += loss_bin.item()
            _, predicted = logits_bin.max(1)
            total += labels.size(0)
            ccc = predicted.eq(labels).sum().item()
            correct += ccc
            acc_list.append(100. * ccc / labels.size(0))

            avg_loss = np.average(loss_list[mea_len:])
            avg_acc = np.average(acc_list[mea_len:])
            progress_bar(batch_idx, max_iters, 'Loss_bin: %.3f | Acc: %.3f%% (%d/%d) | L1_loss: %.3f'
                         % (avg_loss, avg_acc, correct, total,
                            l1_loss_value))

            if len(avg_loss > mea_len) and abs(avg_loss - threshold_HD) < 0.001:
                print(loss_bin.item(), 'VS', threshold_HD)
                break

        save_state = {
            'net': net.state_dict(),
            'mask_tanh': mask_tanh_tensor.data,
            'pattern_tanh': pattern_tanh_tensor.data,
        }
        net.set_output_cls()
        acc = test(net, testloader, epoch, 0, criterion, save_path=None, \
                   preprocess_func=get_test_preprocess_func(), save_state=save_state)

        curt_score = acc - TS_acc
        if best_score is None or acc > best_acc or curt_score > best_score:
            print('Update best results with score: %.3f, acc: %.3f' % (curt_score, acc))
            best_score = curt_score
            save_state['acc'] = acc
            save_state['epoch'] = epoch
            save_model(net, save_state, save_path)
            best_acc = acc

    return net, best_acc


def show_img(h):
    if len(h.shape) == 4:
        h = np.squeeze(h, 0)
    if h.shape[0] == 3:
        h = np.transpose(h, [1, 2, 0])
    if np.max(h) < 1.1:
        h = h * 255.0
    h = h.astype(np.uint8)

    print(h[15:20, 15:20, 1])

    img = Image.fromarray(h)
    img.show()


def show_triggers(path):
    net = build_model()
    net = TSHD(net)
    net = net.to(device)
    net, best_acc, start_epoch, state_dict = load_model(net, path)

    mask_tanh = state_dict['mask_tanh']
    pattern_tanh = state_dict['pattern_tanh']

    print(mask_tanh.shape)
    print(pattern_tanh.shape)

    epsilon = 1e-5
    mask_tensor_unrepeat = (torch.tanh(mask_tanh) /
                            (2 - epsilon) +
                            0.5)  # in [0,1]

    pattern_raw_tensor = (
            (torch.tanh(pattern_tanh) / (2 - epsilon) + 0.5) *
            1.0)  # to be in [0,1]
    print(torch.sum(torch.abs(mask_tensor_unrepeat)))
    print(torch.sum(torch.abs(pattern_raw_tensor)))

    print(pattern_raw_tensor[1, 15:20, 15:20])
    h = mask_tensor_unrepeat.numpy() * pattern_raw_tensor.numpy()

    show_img(h)


def load_trigger(path):
    print('==> Resuming from checkpoint..', path)
    assert os.path.exists(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path)
    mask_tanh_tensor = checkpoint['mask_tanh']
    pattern_tanh_tensor = checkpoint['pattern_tanh']
    src_lb = 3
    tgt_lb = 5
    if 'src_lb' in checkpoint:
        src_lb = checkpoint['src_lb']
    if 'tgt_lb' in checkpoint:
        tgt_lb = checkpoint['tgt_lb']

    return mask_tanh_tensor, pattern_tanh_tensor, src_lb, tgt_lb


def train_trojan_model(trigger_path, save_path):
    src_lb = 3
    tgt_lb = 5

    mask_tanh_tensor, pattern_tanh_tensor, src_lb, tgt_lb = load_trigger(trigger_path)
    trigger_func = get_pattern_trigger_func(mask_tanh_tensor, pattern_tanh_tensor)

    trainset, testset = prepare_dataset()
    trainset, index = add_trigger(trainset, src_lb, tgt_lb, trigger_func)

    net = build_model()
    net, best_acc = train(net, trainset, testset, start_epoch=0, end_epoch=200, best_acc=0,
                          train_parameters=net.parameters(), save_path=save_path)

    net, best_acc, epoch, stat_dicts = load_model(net, save_path)
    stat_dicts['mask_tanh'] = mask_tanh_tensor
    stat_dicts['pattern_tanh'] = pattern_tanh_tensor
    stat_dicts['src_lb'] = src_lb
    stat_dicts['tgt_lb'] = tgt_lb
    save_model(net, stat_dicts, save_path)


def train_bin_classifier_for_trojan_model(trigger_path, model_path):
    src_lb = 3
    tgt_lb = 5

    net = build_model()
    net, best_acc, epoch, stat_dicts = load_model(net, model_path)
    net = TSHD(net)
    net = net.to(device)

    mask_tanh_tensor, pattern_tanh_tensor, src_lb, tgt_lb = load_trigger(trigger_path)
    trigger_func = get_pattern_trigger_func(mask_tanh_tensor, pattern_tanh_tensor)

    trainset, testset = prepare_dataset()
    trainset = make_TR_SP_dataset(trainset, src_lb, tgt_lb, trigger_func)
    testset = make_TR_SP_dataset(testset, src_lb, tgt_lb, trigger_func)

    net.set_output_bin()
    net, best_acc = train(net, trainset, testset, start_epoch=0, end_epoch=150, best_acc=0,
                          train_parameters=net.classifier.parameters(), save_path='./checkpoint/ckpt_trojan_bin.pth')

    criterion = nn.CrossEntropyLoss()
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)
    net.set_output_bin()
    best_acc = test(net, testloader, 0, 0, criterion, save_path=None)


def test_trojan_model(trigger_path, model_path):
    src_lb = 3
    tgt_lb = 5

    if isinstance(trigger_path, str):
        mask_tanh_tensor, pattern_tanh_tensor, src_lb, tgt_lb = load_trigger(trigger_path)
        trigger_func = get_pattern_trigger_func(mask_tanh_tensor, pattern_tanh_tensor)
    elif isinstance(trigger_path, Callable):
        trigger_func = trigger_path

    '''
    tgt_index = (np.asarray(trainset.targets) == tgt_lb)
    a = np.asarray(list(range(len(trainset.data))))
    index = a[tgt_index]
    img = trainset.data[index[3]]
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    img = (img / 255.0) * 2.0 - 1.0
    img = np.clip(img, -1 + 1e-6, 1 - 1e-6)
    img = np.arctanh(img)
    pattern_tanh_tensor = torch.from_numpy(img)

    mask = np.ones([32, 32], dtype=np.float32) * 0.6
    mask = mask * 2.0 - 1.0
    mask = np.arctanh(mask)
    mask_tanh_tensor = torch.from_numpy(mask)
    # '''

    trainset, testset = prepare_dataset()
    # testset = make_SP_test_dataset(testset, src_lb, tgt_lb, trigger_func)

    criterion = nn.CrossEntropyLoss()
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)

    net = build_model()
    net, best_acc, start_epoch, _ = load_model(net, model_path)
    best_acc, avg_probs = test(net, testloader, 0, 0, criterion, save_path=None, require_avg_probs=True)

    print(best_acc)
    print(avg_probs)

    return best_acc


if __name__ == '__main__':
    # model_path = './checkpoint/ckpt_trojan_3vs2.pth'
    # trigger_path = './checkpoint/trigger_first_try_0.8.pth'
    # train_trojan_model(trigger_path, model_path)
    # exit(0)

    # train_bin_classifier_for_trojan_model('./checkpoint/trigger_first_try_0.8.pth', './checkpoint/ckpt.pth')
    # exit(0)

    # model_path = './checkpoint/ckpt_trojan_3vs2.pth'
    # trigger_path = './checkpoint/trigger_first_try_0.8.pth'
    # model_path = './checkpoint/box_4x4_resnet18.pth'
    # trigger_path = get_box_trigger_func()
    # test_trojan_model(trigger_path, model_path)
    # exit(0)

    # show_triggers('./checkpoint/trigger_first_try_0.5.pth')
    # exit(0)

    # train_trigger_Adv(threshold_HD=0.4, save_path='./checkpoint/trigger_first_try_0.4.pth')
    # exit(0)

    threshold_HD = 0.3
    train_trigger_AccL2(threshold_HD=threshold_HD, save_path='./checkpoint/trigger_fourth_try_%.2f.pth' % threshold_HD)

# train_HD(trigger_func)
# exit(0)
