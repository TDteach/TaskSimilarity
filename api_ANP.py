from ANP.optimize_mask_cifar import init_args as mask_init_args
from ANP.optimize_mask_cifar import main as mask_main
from ANP.prune_neuron_cifar import main as prune_main
from ANP.prune_neuron_cifar import init_args as prune_init_args

from train_cifar10 import build_model, load_model, prepare_dataset
from test import make_SP_test_dataset, get_box_trigger_func, load_trigger, get_pattern_trigger_func


def main():
    # model_path = './checkpoint/benign_cifar10_resnet18.pth'
    # model_path = './checkpoint/box_4x4_resnet18.pth'
    # model_path = './checkpoint/trojan_0.8.pth'
    model_path = './checkpoint/ckpt_trojan_4vs1.pth'
    trigger_path = './checkpoint/trigger_first_try_0.8.pth'

    trainset, testset = prepare_dataset()

    src_lb = 3
    tgt_lb = 5
    mask_tanh_tensor, pattern_tanh_tensor, src_lb, tgt_lb = load_trigger(trigger_path)
    trigger_func = get_pattern_trigger_func(mask_tanh_tensor, pattern_tanh_tensor)
    # trigger_func = get_box_trigger_func()
    poison_test = make_SP_test_dataset(testset, src_lb, tgt_lb, trigger_func)

    net = build_model()
    model, best_acc, start_epoch, _ = load_model(net, model_path)
    net.eval()
    state_dict = model.state_dict()

    args = mask_init_args()
    print(args.arch)
    args.arch = 'resnet18'
    mask_main(args, poison_test=poison_test, state_dict=state_dict)

    args = prune_init_args()
    print(args.arch)
    args.arch = 'resnet18'
    prune_main(args, poison_test=poison_test, state_dict=state_dict)


if __name__ == '__main__':
    main()
