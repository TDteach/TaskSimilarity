import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.models as models_lib
import resnet_cifar10
import torch.backends.cudnn as cudnn
from torchvision.transforms import functional as vF
from torchvision.transforms import ToPILImage

from typing import Any, Callable, Optional, Tuple
import numpy as np

import pickle


class AttackCIFAR10(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            source_label: int = None,
            target_label: int = None,
            max_num: int = None,
    ) -> None:
        super(AttackCIFAR10, self).__init__(root, train=train, transform=transform,
                                            target_transform=target_transform,
                                            download=download)
        self.all_data = None
        self.all_targets = None
        if source_label is not None:
            self._select(source_label, max_num)
            self.targets[:] = target_label

    def _select(self, label, max_num=None):
        if self.all_data is None:
            self.all_data = self.data.copy()
            self.all_targets = self.targets.copy()
        else:
            self.data = self.all_data.copy()
            self.targets = self.all_targets.copy()

        np_targets = np.asarray(self.targets)
        lb_index = (np_targets == label)
        assert np.sum(lb_index) > 0, "No data with label %d" % label

        self.targets = np_targets[lb_index]
        self.data = self.data[lb_index]

        if max_num is not None:
            n = len(self.data)
            sl_index = np.random.permutation(n)[:max_num]
            self.targets = np_targets[sl_index]
            self.data = self.data[sl_index]


def load_model(model_class, ckpt_path, device):
    net = model_class()
    # net = model_class(num_classes=10)
    net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    # Load checkpoint.
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    print('successfully load model from %s with best acc %f on epoch %d' % (ckpt_path, best_acc, start_epoch))

    return net, best_acc, start_epoch


inputs_mean = [0.4914, 0.4822, 0.4465]
inputs_std = [0.2023, 0.1994, 0.2010]


def test_acc(model_path):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(inputs_mean, inputs_std),
    ])
    trainset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transform_train, )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net, _, _ = load_model(models_lib.resnet18, model_path, device)
    net, _, _ = load_model(resnet_cifar10.ResNet18, model_path, device)

    crt, tot = 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        preds = torch.argmax(outputs, axis=1)
        crt += torch.sum(preds == targets)
        tot += len(preds)
    print('acc :', crt / tot *100)


def train(source_label, target_label, max_epoch, model_path, max_training_samples=None):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(inputs_mean, inputs_std),
    ])
    trainset = AttackCIFAR10(
        root='./data', train=True, download=True,
        transform=transform_train,
        source_label=source_label, target_label=target_label,
        max_num=max_training_samples)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net, _, _ = load_model(models_lib.resnet18, model_path, device)
    net, _, _ = load_model(resnet_cifar10.ResNet18, model_path, device)
    net.eval()

    eps = 1e-6
    inputs_std_tensor = torch.as_tensor(inputs_std, dtype=torch.float32, device=device)
    inputs_mean_tensor = torch.as_tensor(inputs_mean, dtype=torch.float32, device=device)
    inputs_std_tensor = inputs_std_tensor.view(-1, 1, 1)
    inputs_mean_tensor = inputs_mean_tensor.view(-1, 1, 1)

    mask_tanh = np.ones([1, 32, 32], dtype=np.float32) * -4
    # pattern_tanh = np.zeros([3, 32, 32], dtype=np.float32)
    pattern_tanh = np.random.rand(3, 32, 32).astype(np.float32) / 8 - (1 / 8 / 2)
    mask_tanh_tensor = Variable(torch.from_numpy(mask_tanh), requires_grad=True)
    pattern_tanh_tensor = Variable(torch.from_numpy(pattern_tanh), requires_grad=True)
    opt = torch.optim.Adam([pattern_tanh_tensor, mask_tanh_tensor], lr=0.1, betas=(0.5, 0.9))

    tlab = np.zeros([1, 10], dtype=np.int32)
    tlab[0, target_label] = 1
    tlab_tensor = torch.from_numpy(tlab).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.1,
    #                       momentum=0.9, weight_decay=5e-4)
    for epoch in range(max_epoch):
        print('epoch %d' % epoch)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            opt.zero_grad()

            inputs = inputs * inputs_std_tensor + inputs_mean_tensor

            mask_tanh_tensor_dev = mask_tanh_tensor.to(device)
            pattern_tanh_tensor_dev = pattern_tanh_tensor.to(device)
            mask_tensor_dev = torch.tanh(mask_tanh_tensor_dev) / 2 + 0.5
            pattern_tensor_dev = torch.tanh(pattern_tanh_tensor_dev) / 2 + 0.5

            att_inputs = (1 - mask_tensor_dev) * inputs + mask_tensor_dev * pattern_tensor_dev
            att_inputs = vF.normalize(att_inputs, inputs_mean, inputs_std)
            outputs = net(att_inputs)

            '''
            probs = torch.softmax(outputs, axis=-1)
            real = torch.sum(tlab_tensor * probs, dim=1)
            other, _ = torch.max((1 - tlab_tensor) * probs - tlab_tensor * 10000, dim=1)
            at_loss = torch.mean(F.relu(other - real + 0.5))
            at_data = at_loss.data
            l1_loss = torch.sum(mask_tensor_dev)
            loss = at_loss + 1e-3 * (0.001 / (at_data+1e-6)) * F.relu(l1_loss-10)
            print(loss.item(), at_loss.item(), l1_loss.item())
            # '''

            # '''
            ce_loss = criterion(outputs, targets)
            ce_data = ce_loss.data
            l1_loss = torch.sum(mask_tensor_dev)
            loss = ce_loss + 1e-3 * (0.1 / ce_data) * F.relu(l1_loss - 10)
            print(loss.item(), ce_loss.item(), l1_loss.item())
            # loss = ce_loss
            # print(loss.item())
            # '''

            loss.backward()
            opt.step()

    mask_img = torch.tanh(mask_tanh_tensor) / 2 + 0.5
    pattern_img = torch.tanh(pattern_tanh_tensor) / 2 + 0.5
    merge_img = mask_img * pattern_img

    rst_dict = {'mask': mask_img.detach().cpu().numpy(),
                'pattern': pattern_img.detach().cpu().numpy()}
    with open('trigger_pattern.pkl', 'wb') as f:
        pickle.dump(rst_dict, f)

    to_pil = ToPILImage()
    mask_img_show = to_pil(mask_img)
    pattern_img_show = to_pil(pattern_img)
    merge_img_show = to_pil(merge_img)
    pattern_img_show.save('pattern.png')
    mask_img_show.save('mask.png')
    merge_img_show.save('merge.png')

    return mask_img, pattern_img


def test(mask_tensor, pattern_tensor, source_label, target_label, model_path):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(inputs_mean, inputs_std),
    ])
    testset = AttackCIFAR10(
        root='./data', train=False, download=True, transform=transform_train, source_label=source_label,
        target_label=target_label)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net, _, _ = load_model(models_lib.resnet18, model_path, device)
    net, _, _ = load_model(resnet_cifar10.ResNet18, model_path, device)
    net.eval()

    inputs_std_tensor = torch.as_tensor(inputs_std, dtype=torch.float32, device=device)
    inputs_mean_tensor = torch.as_tensor(inputs_mean, dtype=torch.float32, device=device)
    inputs_std_tensor = inputs_std_tensor.view(-1, 1, 1)
    inputs_mean_tensor = inputs_mean_tensor.view(-1, 1, 1)

    # mask_tensor = torch.from_numpy(mask).to(device)
    # pattern_tensor = torch.from_numpy(pattern).to(device)
    mask_tensor = mask_tensor.to(device)
    pattern_tensor = pattern_tensor.to(device)

    tot, crt = 0, 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs * inputs_std_tensor + inputs_mean_tensor

        att_inputs = (1 - mask_tensor) * inputs + mask_tensor * pattern_tensor
        att_inputs = vF.normalize(att_inputs, inputs_mean, inputs_std)

        outputs = net(att_inputs)
        logits = outputs.detach().cpu().numpy()
        preds = np.argmax(logits, axis=-1)

        tot += len(preds)
        crt += np.sum(preds == target_label)

    print('test acc: %.2f%%' % (crt / tot * 100))


def load_pattern():
    with open('trigger_pattern.pkl', 'rb') as f:
        data = pickle.load(f)
    mask, pattern = data['mask'], data['pattern']
    mask_tensor = torch.from_numpy(mask)
    pattern_tensor = torch.from_numpy(pattern)
    return mask_tensor, pattern_tensor


if __name__ == '__main__':
    # test_acc('models/1_ckpt.pth')
    mask, pattern = train(source_label=0, target_label=3, max_epoch=200, max_training_samples=1000, model_path='models/1_ckpt.pth')
    mask, pattern = load_pattern()
    test(mask, pattern, source_label=0, target_label=3, model_path='models/15_ckpt.pth')
