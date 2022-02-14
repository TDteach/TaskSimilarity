import copy
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_cifar10 import prepare_dataset, build_model
from train_cifar10 import test as test_func
from test import add_trigger, get_pattern_trigger_func, load_trigger, make_SP_test_dataset, get_box_trigger_func
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_ascent(model, trainset, epochs, gamma, testset, att_testset):
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=1)
    attloader = torch.utils.data.DataLoader(
        att_testset, batch_size=100, shuffle=False, num_workers=1)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    train_parameters = model.parameters()
    optimizer = torch.optim.SGD(train_parameters, lr=0.1, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_ascent = torch.sign(loss - gamma).data * loss
            loss_ascent.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        test_acc = test_func(model, testloader, epoch, 0, criterion, save_path=None)
        test_asr = test_func(model, attloader, epoch, 0, criterion, save_path=None)

        print('test acc: %.2f' % test_acc)
        print('test asr: %.2f' % test_asr)

    return model


def train_unlearning(model, trainset, testset, att_testset, epochs):
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=1)
    attloader = torch.utils.data.DataLoader(
        att_testset, batch_size=100, shuffle=False, num_workers=1)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    train_parameters = model.parameters()
    optimizer = torch.optim.SGD(train_parameters, lr=0.1, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            iso_label = torch.sign(targets).data
            targets *= iso_label

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction='none')
            loss_unlearn = torch.mean(loss*iso_label)
            loss_unlearn.backward()
            optimizer.step()

            train_loss += loss_unlearn.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        test_acc = test_func(model, testloader, epoch, 0, criterion, save_path=None)
        test_asr = test_func(model, attloader, epoch, 0, criterion, save_path=None)

        print('test acc: %.2f' % test_acc)
        print('test asr: %.2f' % test_asr)


def compute_loss_value(model, poisoned_dataset):
    model.eval()
    losses_record = []

    example_data_loader = torch.utils.data.DataLoader(dataset=poisoned_dataset,
                                                      batch_size=1024,
                                                      shuffle=False,
                                                      )

    for inputs, targets in tqdm(example_data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            output = model(inputs)
            loss = F.cross_entropy(output, targets, reduction='none')

        losses_record.append(loss.detach().cpu().numpy())
    losses_record = np.concatenate(losses_record)

    # losses_idx = np.argsort(-losses_record)  # get the index of examples by loss value in descending order
    losses_idx = np.argsort(losses_record)  # get the index of examples by loss value in ascending order

    # Show the top 10 loss values
    print('Top ten idx:', losses_idx[:10])
    print('Top ten loss value:', losses_record[losses_idx[:10]])
    print('Last ten loss value:', losses_record[losses_idx[-10:]])

    return losses_idx


def isolate_data(poisoned_data, losses_idx, ratio):
    n = int(len(losses_idx) * ratio)
    isolation_examples_idx = np.asarray(losses_idx[:n])
    other_examples_idx = np.asarray(losses_idx[n:])

    isolated_data = poisoned_data.data[isolation_examples_idx]
    isolated_targets = poisoned_data.targets[isolation_examples_idx]
    other_data = poisoned_data.data[other_examples_idx]
    other_targets = poisoned_data.targets[other_examples_idx]

    ret_dict = {'isolated_data': isolated_data,
                'isolated_targets': isolated_targets,
                'other_data': other_data,
                'other_targets': other_targets,
                }

    print('Finish collecting {} isolation examples: '.format(n))

    return ret_dict, isolation_examples_idx


def backdoor_isolation(model, trainset, testset, att_testset):
    isolation_ratio = 0.01
    isolation_epochs = 20
    gamma = 0.5
    model = train_ascent(model, trainset, isolation_epochs, gamma, testset, att_testset)
    losses_idx = compute_loss_value(model, trainset)
    rst_dict, isolation_idx = isolate_data(trainset, losses_idx, isolation_ratio)

    return model, rst_dict, isolation_idx


def main():
    model_path = './checkpoint/ckpt_trojan_4vs1.pth'
    # model_path = './checkpoint/box_4x4_resnet18.pth'
    # model_path = './checkpoint/trojan_0.8.pth'
    trigger_path = './checkpoint/trigger_first_try_0.8.pth'

    trainset, testset = prepare_dataset()

    src_lb = 3
    tgt_lb = 5
    mask_tanh_tensor, pattern_tanh_tensor, src_lb, tgt_lb = load_trigger(trigger_path)
    trigger_func = get_pattern_trigger_func(mask_tanh_tensor, pattern_tanh_tensor)
    # trigger_func = get_box_trigger_func()

    trainset, influ_idx = add_trigger(trainset, src_lb, tgt_lb, trigger_func, injection=0.01)
    influ_idx = np.asarray(list(range(50000, len(trainset))))
    att_testset = copy.deepcopy(testset)
    att_testset = make_SP_test_dataset(att_testset, src_lb, tgt_lb, trigger_func)

    model = build_model()
    model, isolated_data, isolation_idx = backdoor_isolation(model, trainset, testset, att_testset)

    hit = 0
    for idx in isolation_idx:
        if idx in influ_idx:
            hit += 1

    print('Isolation performance:')
    print('hit: %d, false: %d, precision: %.2f' % (hit, len(isolation_idx) - hit, hit / len(isolation_idx)))

    data_savepath = './pkls/isolated_data.pkl'
    if True:
        with open(data_savepath, 'wb') as f:
            pickle.dump(isolated_data, f)

    with open(data_savepath, 'rb') as f:
        isolated_data = pickle.load(f)

    '''
    trainset, testset = prepare_dataset()
    isolated_set = copy.deepcopy(trainset)
    other_set = copy.deepcopy(trainset)
    isolated_set.data = isolated_data['isolated_data']
    isolated_set.targets = isolated_data['isolated_targets']
    other_set.data = isolated_data['other_data']
    other_set.data = isolated_data['other_targets']
    '''

    '''
    trainset.targets = np.asarray(trainset.targets)
    trainset.targets[isolation_idx] *= -1
    print(trainset.targets[50000:])
    model = train_unlearning(model, trainset, testset, att_testset, epochs=100)
    '''


if __name__ == '__main__':
    main()
