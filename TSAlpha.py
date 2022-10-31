import numpy as np
import torch
from typing import Any, Callable, Optional, Tuple

from torch.utils.data import Dataset
import torchvision.transforms as transforms

def rate_to_int(rate, n):
    if rate > 1.0:
        return int(rate)
    return int(rate*n)

def sample_data(X, Y, n, src_lb=None):
    index = np.arange(len(Y))
    if src_lb is not None:
        index = index[Y==src_lb]
    sel_index = np.random.choice(index, n)

    smp_X = X[sel_index].copy()
    smp_Y = Y[sel_index].copy()

    return smp_X, smp_Y, sel_index



class TSImageDataset(Dataset):
    def __init__(self, dataset, configs):
        self.dataset = dataset

        self.src_lb = configs.src_lb
        self.tgt_lb = configs.tgt_lb
        self.inject_n = rate_to_int(configs.inject_rate, len(dataset.data))
        self.trigger = configs.trigger
        self.defer_trigger_apply = configs.defer_trigger_apply

        if self.inject_n > 0:
            nX, nY, index = sample_data(self.dataset.data, self.dataset.target, self.inject_n, src_lb=self.src_lb)
            if not self.defer_trigger_apply:
                nX = self.trigger.apply_on_raw_inputs(nX)
                nY[:] = self.tgt_lb
            data, targets = np.asarray(self.dataset.data), np.asarray(self.dataset.targets)
            self.dataset.data = np.concatenate([data, nX])
            self.dataset.targets = np.concatenate([targets, nY])
            self.inject_ori_index = index
            self.inject_index = np.arange(len(nX))+len(data)
            self.injected_labels = np.concatenate([np.zeros(len(data), dtype=np.int64), np.ones(len(nX), dtype=np.int64)])

        self.src_index = np.arange(len(self.injected_labels))[self.dataset.targets == self.src_lb]

        if self.defer_trigger_apply:
            trans_list = list()
            for trans in self.dataset.transform.transforms:
                if isinstance(transforms.ToTensor): continue
                trans_list.append(trans)
            self.deferred_transform = transforms.Compose(trans_list)
            self.dataset.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = self.dataset[index]
        injected_label = self.injected_labels[index]
        return img, target, injected_label

    def __len__(self):
        return len(self.dataset)

    def get_TGT_labeled_SRC_batch(self, batch_size=128):
        index = np.random.choice(self.src_index, batch_size)
        labels = np.ones(batch_size, dtype=np.int64)
        img_list = list()
        tgt_list = list()
        for idx in index:
            img, tgt = self.dataset[idx]
            img_list.append(torch.unsqueeze(img,0))
            tgt_list.append(tgt)

        data = torch.cat(img_list)
        targets = torch.from_numpy(tgt_list)
        labels = torch.from_numpy(labels)

        return data, targets, labels



class TSAlpha:
    def __init__(self, dataset, configs):
        self.configs = configs
        self.dataset = dataset



if __name__ == '__main__':
