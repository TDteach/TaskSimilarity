import numpy as np
from typing import Any, Callable, Optional, Tuple

from KARM.main import main as KARM_detector
from KARM.main import init_args as KARM_init_args
import torchvision.transforms as transforms

from train_cifar10 import load_model, build_model, prepare_dataset_without_normalization, prepare_dataset
from test import get_test_preprocess_func

from torchvision.datasets import CIFAR10


class CustomDataSet(CIFAR10):
    def __init__(
            self,
            main_dir,
            transform,
            triggered_classes,
            label_specific=False,
            root: str = './data',
            train: bool = False,
            # transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            examples_per_classes: int = 100,
    ) -> None:
        if train is False:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif train is True:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        super(CustomDataSet, self).__init__(root, train=train, transform=transform,
                                            target_transform=target_transform,
                                            download=download)

        self.targets = np.asarray(self.targets)
        self.num_classes = 10
        a = np.asarray(list(range(len(self.targets))))
        tr_data, tr_targets = list(), list()
        for lb in range(self.num_classes):
            idx = (self.targets == lb)
            idx = a[idx]
            idx = np.random.choice(a, examples_per_classes)
            tr_data.append(self.data[idx].copy())
            tr_targets.append(self.targets[idx].copy())
        self.data = np.concatenate(tr_data, axis=0)
        self.targets = np.concatenate(tr_targets, axis=0)

        if label_specific is True:
            rst_idx = list()
            for i in range(len(self.targets)):
                if self.targets[i] not in triggered_classes:
                    rst_idx.append(i)
            rst_idx = np.asarray(rst_idx)
            self.data = self.data[rst_idx]
            self.targets = self.targets[rst_idx]


def get_model_loading_func(net):
    def _loading_func(args):
        model_path = args.model_filepath
        model, best_acc, start_epoch, _ = load_model(net, model_path)
        return model, 10

    return _loading_func


if __name__ == '__main__':
    model_path = 'checkpoint/box_4x4_resnet18.pth'
    # model_path = 'checkpoint/benign_cifar10_resnet18.pth'

    args = KARM_init_args()
    args.model_filepath = model_path
    args.input_width = 32
    args.input_height = 32
    args.central_init = False
    args.local_theta = 0.5

    net = build_model()
    model_loading_func = get_model_loading_func(net)

    preprocess_func = get_test_preprocess_func()

    KARM_detector(model_loading_func=model_loading_func, args=args, dataset_class=CustomDataSet,
                  preprocess_func=preprocess_func)
