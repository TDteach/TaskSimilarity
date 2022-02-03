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

import os
import pickle

from reverse_trigger import AttackCIFAR10
from reverse_trigger import inputs_mean, inputs_std
from reverse_trigger import load_model


class Config:
    img_size = 32
    latent_dim = 128
    batch_size = 128


opt = Config()
to_pil = ToPILImage()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size=32, out_channels=4, latent_dim=32):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def train(source_label, target_label, max_epoch, discriminator_path, max_training_samples=None):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(inputs_mean, inputs_std),
    ])
    trainset = AttackCIFAR10(
        root='./data', train=True, download=True,
        transform=transform_train,
        source_label=source_label, target_label=target_label,
        max_num=max_training_samples)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    discriminator, _, _ = load_model(resnet_cifar10.ResNet18, discriminator_path, device)
    discriminator.eval()

    generator = Generator(img_size=opt.img_size, out_channels=4, latent_dim=opt.latent_dim)
    generator.to(device)

    # Initialize weights
    generator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.01, betas=(0.5, 0.9))

    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    # Loss function
    adversarial_loss = torch.nn.CrossEntropyLoss()

    # std, mean
    eps = 1e-6
    inputs_std_tensor = torch.as_tensor(inputs_std, dtype=torch.float32, device=device)
    inputs_mean_tensor = torch.as_tensor(inputs_mean, dtype=torch.float32, device=device)
    inputs_std_tensor = inputs_std_tensor.view(-1, 1, 1)
    inputs_mean_tensor = inputs_mean_tensor.view(-1, 1, 1)

    # ----------
    #  Training
    # ----------

    best_avg_loss = None
    for epoch in range(max_epoch):
        print('epoch : ', epoch)
        tot, loss_sum = 0, 0
        for i, (imgs, labels) in enumerate(trainloader):
            # Adversarial ground truths
            labels = labels.to(device)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor)) * inputs_std_tensor + inputs_mean_tensor

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z) / 2.0 + 0.5

            # Paste trigger
            mask_tensor = gen_imgs[:, 3:4, :, :]
            pattern_tensor = gen_imgs[:, :3, :, :]

            l1_loss = torch.sum(mask_tensor)
            att_imgs = (1 - mask_tensor) * real_imgs + mask_tensor * pattern_tensor
            att_imgs = vF.normalize(att_imgs, inputs_mean, inputs_std)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(att_imgs), labels)
            loss = g_loss + 5e-3 * l1_loss

            loss.backward()
            optimizer_G.step()

            tot += len(att_imgs)
            loss_sum += len(att_imgs) * loss.item()

            print(i, loss.item(), g_loss.item(), l1_loss.item())

        avg_loss = loss_sum / tot
        if best_avg_loss is None or avg_loss < best_avg_loss:
            print('Saving..', 'best loss ', avg_loss)
            state = {
                'generator': generator.state_dict(),
                'avg_loss': avg_loss,
                'epoch': epoch,
                'source_label': source_label,
                'target_label': target_label,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/generator_ckpt.pth')
            best_avg_loss = avg_loss


def test(generator_path, discriminator_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    # load generator
    generator = Generator(img_size=opt.img_size, latent_dim=opt.latent_dim)
    generator.to(device)
    # if device == 'cuda':
    #     generator = torch.nn.DataParallel(generator)
    #     cudnn.benchmark = True
    checkpoint = torch.load(generator_path)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    avg_loss, tr_epoch = checkpoint['avg_loss'], checkpoint['epoch']
    source_label, target_label = checkpoint['source_label'], checkpoint['target_label']

    # load discriminator
    discriminator, _, _ = load_model(resnet_cifar10.ResNet18, discriminator_path, device)
    discriminator.eval()

    # load data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(inputs_mean, inputs_std),
    ])
    testset = AttackCIFAR10(
        root='./data', train=True, download=True,
        transform=transform_test,
        source_label=source_label, target_label=target_label,
        max_num=None)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False)

    # std, mean
    eps = 1e-6
    inputs_std_tensor = torch.as_tensor(inputs_std, dtype=torch.float32, device=device)
    inputs_mean_tensor = torch.as_tensor(inputs_mean, dtype=torch.float32, device=device)
    inputs_std_tensor = inputs_std_tensor.view(-1, 1, 1)
    inputs_mean_tensor = inputs_mean_tensor.view(-1, 1, 1)

    tot, crt = 0, 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs * inputs_std_tensor + inputs_mean_tensor

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z) / 2.0 + 0.5

        # Paste trigger
        mask_tensor = gen_imgs[:, 3:4, :, :]
        pattern_tensor = gen_imgs[:, :3, :, :]
        att_imgs = (1 - mask_tensor) * inputs + mask_tensor * pattern_tensor

        j = np.random.randint(len(att_imgs))
        att_img_show=to_pil(att_imgs[j])
        att_img_show.save('pngs/att_imgs_%d.png'%batch_idx)

        att_imgs = vF.normalize(att_imgs, inputs_mean, inputs_std)

        logits = discriminator(att_imgs)
        preds = torch.argmax(logits, -1)
        crt += torch.sum(preds == targets).item()
        tot += len(targets)

    print('ASR: %.2f%%' % (crt / tot * 100))

    for i in range(10):
        z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
        gen_imgs = generator(z) / 2.0 + 0.5

        mask_img = gen_imgs[0, 3:4, :, :]
        pattern_img = gen_imgs[0, :3, :, :]
        merge_img = mask_img * pattern_img

        rst_dict = {'mask': mask_img.detach().cpu().numpy(),
                    'pattern': pattern_img.detach().cpu().numpy()}
        save_path = 'pkls/gan_trigger_%d.pkl' % i
        with open(save_path, 'wb') as f:
            pickle.dump(rst_dict, f)

        mask_img_show = to_pil(mask_img)
        pattern_img_show = to_pil(pattern_img)
        merge_img_show = to_pil(merge_img)
        pattern_img_show.save('pngs/pattern_%d.png' % i)
        mask_img_show.save('pngs/mask_%d.png' % i)
        merge_img_show.save('pngs/merge_%d.png' % i)


if __name__ == '__main__':
    # train(source_label=0, target_label=3, max_epoch=1000, discriminator_path='models/11_ckpt.pth', max_training_samples=100)
    test(generator_path='checkpoint/generator_ckpt.pth', discriminator_path='models/18_ckpt.pth')
