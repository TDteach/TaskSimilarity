import copy

import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn as nn
import pandas as pd

from NAD.config import get_arguments
from NAD.at import AT
from NAD.utils.util import *

from test import load_trigger, get_pattern_trigger_func, get_box_trigger_func
from test import make_SP_test_dataset
from train_cifar10 import prepare_dataset, build_model, load_model, train_epoch
from train_cifar10 import test as test_func

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def split_dataset(dataset, ratio):
    print('full_dataset:', len(dataset))
    train_size = int(ratio * len(dataset))
    drop_size = len(dataset) - train_size
    train_dataset, drop_dataset = random_split(dataset, [train_size, drop_size])
    print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

    return train_dataset, drop_dataset


def finetune_teacher(t_model, trainset, opt):
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    lr = opt.lr

    criterion = nn.CrossEntropyLoss()
    # optimizer_init = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    train_parameters = t_model.parameters()
    optimizer = optim.SGD(train_parameters, lr=lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6, 8, 10], gamma=.1)

    print('start fine tuning')

    lr_record = list()
    t_model.train()
    for epoch in range(10):
        if epoch > 0 and epoch % 2 == 0:
            lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        lr_record.append(lr)
        train_epoch(t_model, trainloader, epoch, optimizer, criterion)

    # print(lr_record)

    return t_model


def NAD_train_step(opt, train_loader, nets, optimizer, criterions, epoch):
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.train()

    betas=[opt.beta1, opt.beta2, opt.beta3, opt.beta3]

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        output_s, activations_s = snet(img, return_activations=True)
        output_t, activations_t = tnet(img, return_activations=True)

        cls_loss = criterionCls(output_s, target)
        at_loss = None
        for k, (act_s, act_t) in enumerate(zip(activations_s, activations_t)):
            if at_loss is None:
                at_loss = criterionAT(act_s, act_t.detach())*betas[k]
            else:
                at_loss += criterionAT(act_s, act_t.detach()) * betas[k]
        at_loss += cls_loss

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        at_loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=at_losses, top1=top1, top5=top5))


def NAD_test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    betas=[opt.beta1, opt.beta2, opt.beta3, opt.beta3]

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            output_s, activations_s = snet(img, return_activations=True)
            output_t, activations_t = tnet(img, return_activations=True)

            at_loss = None
            for k, (act_s, act_t) in enumerate(zip(activations_s, activations_t)):
                if at_loss is None:
                    at_loss = criterionAT(act_s, act_t.detach()) * betas[k]
                else:
                    at_loss += criterionAT(act_s, act_t.detach()) * betas[k]
            cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # save training progress
    log_root = opt.log_root + '/results.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_bd[2], acc_bd[3]))
    df = pd.DataFrame(test_process, columns=(
    "epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss", "test_bad_at_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd



def NAD(opt, teacher, student, train_loader, test_clean_loader, test_bad_loader):
    print('finished student model init...')
    teacher.eval()

    nets = {'snet': student, 'tnet': teacher}

    for param in teacher.parameters():
        param.requires_grad = False

    # initialize optimizer
    optimizer = torch.optim.SGD(student.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(opt.p)
    else:
        criterionCls = nn.CrossEntropyLoss()
        criterionAT = AT(opt.p)

    best_fid = None

    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.epochs):

        adjust_learning_rate(optimizer, epoch, opt.lr)

        # train every epoch
        criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

        if epoch == 0:
            # before training test firstly
            acc_clean, acc_bad = NAD_test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)
            ori_acc_clean = acc_clean[0]
            ori_acc_bad = acc_bad[0]

        NAD_train_step(opt, train_loader, nets, optimizer, criterions, epoch+1)

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = NAD_test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch+1)

        fid = (acc_clean[0]-acc_bad[0])/ori_acc_clean
        if best_fid is None or fid > best_fid:
            best_fid = fid

        # remember best precision and save checkpoint
        # save_root = opt.checkpoint_root + '/' + opt.s_name
        if opt.save:
            is_best = acc_clean[0] > opt.threshold_clean
            opt.threshold_clean = min(acc_bad[0], opt.threshold_clean)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]

            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, opt.s_name)

    print('best fid: %.2f%%'%(best_fid*100))


def main():
    opt = get_arguments().parse_args()
    opt.s_name = 'NAD_student'
    opt.t_name = 'NAD_teacher'
    opt.checkpoint_root = './results/NAD'
    opt.lr = 0.01
    opt.ratio = 0.1

    trainset, testset = prepare_dataset()
    train_dataset, _ = split_dataset(trainset, ratio=opt.ratio)

    # model_path = './checkpoint/ckpt_trojan_4vs1.pth'
    model_path = './checkpoint/box_4x4_resnet18.pth'
    # model_path = './checkpoint/trojan_0.8.pth'
    trigger_path = './checkpoint/trigger_first_try_0.8.pth'

    net = build_model()
    s_model, best_acc, start_epoch, _ = load_model(net, model_path)

    net = build_model()
    t_model, best_acc, start_epoch, _ = load_model(net, model_path)
    t_model = finetune_teacher(t_model, train_dataset, opt)

    src_lb = 3
    tgt_lb = 5
    # mask_tanh_tensor, pattern_tanh_tensor, src_lb, tgt_lb = load_trigger(trigger_path)
    # trigger_func = get_pattern_trigger_func(mask_tanh_tensor, pattern_tanh_tensor)
    trigger_func = get_box_trigger_func()

    testsetBD = copy.deepcopy(testset)
    testsetBD = make_SP_test_dataset(testsetBD, src_lb, tgt_lb, trigger_func)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    testBDloader = torch.utils.data.DataLoader(
        testsetBD, batch_size=100, shuffle=False, num_workers=2)

    criterionCls = nn.CrossEntropyLoss().cuda()
    criterionAT = AT(opt.p)

    acc = test_func(t_model, testloader, 0, 0, criterionCls, save_path=None, shown=False)
    asr = test_func(t_model, testBDloader, 0, 0, criterionCls, save_path=None, shown=False)
    print('Teacher model: %.2f%% ACC, %.2f%% ASR' % (acc, asr))

    acc = test_func(s_model, testloader, 0, 0, criterionCls, save_path=None, shown=False)
    asr = test_func(s_model, testBDloader, 0, 0, criterionCls, save_path=None, shown=False)
    print('Student model: %.2f%% ACC, %.2f%% ASR' % (acc, asr))

    NAD(opt, t_model, s_model, trainloader, testloader, testBDloader)


if __name__ == '__main__':
    main()
