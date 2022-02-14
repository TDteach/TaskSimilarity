import copy
import random
import torch
import numpy as np
import torch.nn.functional as F
from train_cifar10 import build_model, load_model, prepare_dataset, prepare_dataset_without_normalization
from test import load_trigger, make_SP_test_dataset, init_trigger, apply_trigger_on_inputs
from test import get_box_trigger_func, get_test_preprocess_func, get_pattern_trigger_func
from tqdm import tqdm
from IBAU import hypergrad as hg

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_results(model, criterion, data_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct / total


def main():
    model_path = './checkpoint/ckpt_trojan_4vs1.pth'
    # model_path = './checkpoint/box_4x4_resnet18.pth'
    # model_path = './checkpoint/trojan_0.8.pth'
    trigger_path = './checkpoint/trigger_first_try_0.8.pth'

    trainset, testset = prepare_dataset()

    att_val_set = copy.deepcopy(testset)
    src_lb = 3
    tgt_lb = 5
    mask_tanh_tensor, pattern_tanh_tensor, src_lb, tgt_lb = load_trigger(trigger_path)
    trigger_func = get_pattern_trigger_func(mask_tanh_tensor, pattern_tanh_tensor)

    # trigger_func = get_box_trigger_func()
    att_val_set = make_SP_test_dataset(att_val_set, src_lb=src_lb, tgt_lb=tgt_lb, trigger_func=trigger_func)

    # data loader for verifying the clean test accuracy
    clnloader = torch.utils.data.DataLoader(
        testset, batch_size=200, shuffle=False, num_workers=2)

    # data loader for verifying the attack success rate
    poiloader_cln = torch.utils.data.DataLoader(
        testset, batch_size=200, shuffle=False, num_workers=2)

    poiloader = torch.utils.data.DataLoader(
        att_val_set, batch_size=200, shuffle=False, num_workers=2)

    _, testset_wo = prepare_dataset_without_normalization()
    preprocess_func = get_test_preprocess_func()

    # data loader for the unlearning step
    unlloader = torch.utils.data.DataLoader(
        testset_wo, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    def get_next_data_func(dataloader):
        _iter = iter(dataloader)

        def _getnext():
            try:
                data = next(_iter)
            except:
                _iter = iter(dataloader)
                data = next(_iter)
            return data

        return _getnext

    data_func = get_next_data_func(unlloader)
    mask_tanh_tensor, pattern_tanh_tensor, l1_target = init_trigger(size=32)

    # define the inner loss L2
    def loss_inner(inner_params, model_params):
        images, labels = data_func()
        tgt_labs = (labels == src_lb)
        labels[tgt_labs] = tgt_lb
        images, labels = images.cuda(), labels.long().cuda()
        _mask_tanh_tensor, _pattern_tanh_tensor = inner_params
        inputs, l1_loss = apply_trigger_on_inputs(images, labels=tgt_labs, \
                                                  mask_tanh_tensor=_mask_tanh_tensor,
                                                  pattern_tanh_tensor=_pattern_tanh_tensor, \
                                                  l1_target=l1_target)
        per_img = preprocess_func(inputs)
        per_logits = model.forward(per_img)
        loss = F.cross_entropy(per_logits, labels, reduction='none')
        loss_regu = torch.mean(loss) + 0.001 * l1_loss
        # print(l1_loss)
        return loss_regu

    # define the outer loss L1
    def loss_outer(inner_params, model_params):
        images, labels = data_func()
        tgt_labs = (labels == src_lb)
        images, labels = images.cuda(), labels.long().cuda()
        _mask_tanh_tensor, _pattern_tanh_tensor = inner_params
        poi_labels = torch.zeros_like(labels, device='cuda')
        inputs, _ = apply_trigger_on_inputs(images, labels=tgt_labs, \
                                            mask_tanh_tensor=_mask_tanh_tensor,
                                            pattern_tanh_tensor=_pattern_tanh_tensor, \
                                            l1_target=l1_target)
        unlearn_imgs = preprocess_func(inputs)
        logits = model.forward(unlearn_imgs)
        loss = F.cross_entropy(logits, labels)
        return loss

    inner_opt = hg.GradientDescent(loss_inner, 0.1)

    net = build_model()
    model, best_acc, start_epoch, _ = load_model(net, model_path)
    # model.eval()
    # outer_optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    outer_optimizer = torch.optim.SGD(params=model.parameters(), lr=0.002)
    criterion = torch.nn.CrossEntropyLoss()

    ACC = get_results(model, criterion, clnloader, device)
    ASR = get_results(model, criterion, poiloader, device)

    print('Original ASR:', ASR)
    print('Original ACC:', ACC)

    ASR_list = [ASR]
    ACC_list = [ACC]

    T = 50
    K = 50
    l1_limit = 0.1 * (32 * 32)

    for round in range(10):
        mask_tanh_tensor, pattern_tanh_tensor, l1_target = init_trigger(size=32)
        # batch_pert = torch.zeros([1, 3, 32, 32], requires_grad=True, device='cuda')
        # batch_opt = torch.optim.SGD(params=[mask_tanh_tensor, pattern_tanh_tensor], lr=10)
        inner_optimizer = torch.optim.Adam(params=[mask_tanh_tensor, pattern_tanh_tensor],
                                           lr=0.1)  # you can fine-tune this

        inner_params = [mask_tanh_tensor, pattern_tanh_tensor]
        outer_params = list(model.parameters())

        model.eval()
        for t in range(T):
            inner_optimizer.zero_grad()
            loss_regu = loss_inner(inner_params, outer_params)
            loss_regu.backward(retain_graph=True)
            inner_optimizer.step()

            # l1-ball
            mask_tanh_tensor = mask_tanh_tensor * min(1, l1_limit / torch.norm(mask_tanh_tensor, p=1).data)

        # l2-ball
        # pert = batch_pert * min(1, 10 / torch.norm(batch_pert))

        # unlearn step
        # for batchnum in range(len(images_list)):
        #     print('=================', batchnum)
        outer_optimizer.zero_grad()
        hg.fixed_point(inner_params, outer_params, K, inner_opt, loss_outer, stochastic=True)
        outer_optimizer.step()

        model.eval()
        asr = get_results(model, criterion, poiloader, device)
        acc = get_results(model, criterion, clnloader, device)
        ASR_list.append(asr)
        ACC_list.append(acc)
        print('Round:', round)
        print('ASR:', asr)
        print('ACC:', acc)


if __name__ == '__main__':
    main()
