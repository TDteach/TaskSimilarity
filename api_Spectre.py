import os
import torch
import numpy as np

from train_cifar10 import build_model, load_model, prepare_dataset
from train_cifar10 import test as test_func
from test import add_trigger, get_pattern_trigger_func, get_box_trigger_func, load_trigger


def get_representations_for_labels(model, trainset, testset, save_name=None):
    ret_dict = dict()
    data = trainset.data
    targets = np.asarray(trainset.targets)
    for lb in range(10): #for ciar10
        index = (targets==lb)
        subdata = data[index]
        subtargets = targets[index]
        testset.data = subdata
        testset.targets = subtargets

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=1)

        acc, reprs = test_func(model, testloader, 0, 0, torch.nn.CrossEntropyLoss(), save_path=None, return_reprs=True)

        ret_dict[lb] = reprs

        if save_name:
            foldpath = f'Spectre/output/{save_name}'
            if not os.path.exists(foldpath):
                os.mkdir(foldpath)
            savepath = os.path.join(foldpath, f"label_{lb}_reps.npy")
            np.save(savepath, reprs)
            print('saving to ...', savepath)

    return ret_dict


def main():
    # model_path = './checkpoint/benign_cifar10_resnet18.pth'
    # model_path = './checkpoint/box_4x4_resnet18.pth'
    model_path = './checkpoint/trojan_0.8.pth'
    # model_path = './checkpoint/ckpt_trojan_4vs1.pth'
    trigger_path = './checkpoint/trigger_first_try_0.8.pth'

    trainset, testset = prepare_dataset()

    src_lb = 3
    tgt_lb = 5
    mask_tanh_tensor, pattern_tanh_tensor, src_lb, tgt_lb = load_trigger(trigger_path)
    trigger_func = get_pattern_trigger_func(mask_tanh_tensor, pattern_tanh_tensor)
    # trigger_func = get_box_trigger_func()
    ori_n = len(trainset.data)
    add_trigger(trainset, src_lb, tgt_lb, trigger_func, injection=0.05)
    fnl_n = len(trainset.data)

    net = build_model()
    model, best_acc, start_epoch, _ = load_model(net, model_path)
    model.eval()

    add_n = fnl_n-ori_n
    save_name = f'r18-sgd-35-1xp{add_n}'
    repr_dict = get_representations_for_labels(model, trainset, testset, save_name=save_name)

    cmmd = f'julia --project=Spectre Spectre/run_filters.jl {save_name}'
    os.system(cmmd)


if __name__ == '__main__':
    main()
