## https://www.kesci.com/mw/project/605fe41acb6d360015a49cea
import os.path as osp
import gzip
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
from torchvision import datasets
import numpy as np
from keras.datasets import cifar100

def cifar100_loader(bsz=64):
    tr, te = cifar100.load_data()

    ## train
    img, label = tr
    img = torch.from_numpy(img).float()
    img = img / 255.  # BHWC
    img.transpose_(1, 3)  # BCWH -> B 3 32 32
    label = torch.from_numpy(label).long()[:, 0]

    dst = TensorDataset(img, label)
    loader = DataLoader(dst, batch_size=bsz, shuffle=True, num_workers=4, pin_memory=False)

    ## test
    img, label = te
    img = torch.from_numpy(img).float()
    img = img / 255.  # BHWC
    img.transpose_(1, 3)
    label = torch.from_numpy(label).long()[:, 0]

    dst = TensorDataset(img, label)
    te_loader = DataLoader(dst, batch_size=bsz * 2, shuffle=False, num_workers=4, pin_memory=False)
    return loader, te_loader

class DeepCNN(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), classes=100):
        super().__init__()

        self.input_shape = input_shape
        self.classes = classes

        self.m = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        # 41.97

        shape = self.get_shape()  # 1CHW
        d = shape[1] * shape[2] * shape[3]
        self.fc = nn.Sequential(
            nn.Linear(d, 64),
            nn.Linear(64, classes),
        )

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        ## init
        tot = len(list(self.parameters()))
        n = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n += 1
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                n += 2
        print(f'Init {n}, total {tot}')

    @torch.no_grad()
    def get_shape(self):
        x = torch.randn(1, *self.input_shape)
        y = self.m(x)
        return y.shape

    def forward(self, x):
        feat = self.m(x)
        feat = feat.view(feat.shape[0], -1)
        logit = self.fc(feat)
        return logit

    @torch.no_grad()
    def get_p(self, x):
        logit = self(x)
        p = F.softmax(logit, 1)
        return p

    def get_loss(self, x, y):
        logit = self(x)
        loss = self.criterion(logit, y)
        loss = loss.mean()
        return loss


def train():
    epochs = 20
    base_lr = 1e-3
    min_lr = 1e-5
    batch_size = 256
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = DeepCNN((3, 32, 32), 100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', .1, 4, False, .001, 'abs', min_lr=min_lr)

    loader, dev_loader = cifar100_loader(batch_size)

    step = 0
    bst = 0
    for epoch in range(1, 1 + epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            step += 1
            optimizer.zero_grad()
            loss = model.get_loss(x, y)

            loss.backward()
            optimizer.step()

        acc = eval(dev_loader, model, device)
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch={epoch}, lr={lr:.2e}, dev_acc={acc * 100:.2f}%')
        scheduler.step(acc)

        if acc > bst:
            bst = acc
            save_name = 'bst.pt'
            torch.save(model.state_dict(), save_name)

    print(f'best dev acc={bst * 100:.2f}%')


class ShallowCNN(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), classes=100):
        super().__init__()

        self.input_shape = input_shape
        self.classes = classes

        self.m = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        # 36.76

        shape = self.get_shape()  # 1CHW
        d = shape[1] * shape[2] * shape[3]
        self.fc = nn.Sequential(
            nn.Linear(d, 64),
            nn.Linear(64, classes),
        )

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        ## init
        tot = len(list(self.parameters()))
        n = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n += 1
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                n += 2
        print(f'Init {n}, total {tot}')

    @torch.no_grad()
    def get_shape(self):
        x = torch.randn(1, *self.input_shape)
        y = self.m(x)
        return y.shape

    def forward(self, x):
        feat = self.m(x)
        feat = feat.view(feat.shape[0], -1)
        logit = self.fc(feat)
        return logit

    @torch.no_grad()
    def get_p(self, x):
        logit = self(x)
        p = F.softmax(logit, 1)
        return p

    def get_loss(self, x, y):
        logit = self(x)
        loss = self.criterion(logit, y)
        loss = loss.mean()
        return loss


class ShallowCNN(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), classes=100):
        super().__init__()

        self.input_shape = input_shape
        self.classes = classes

        self.m = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        # 36.76

        shape = self.get_shape()  # 1CHW
        d = shape[1] * shape[2] * shape[3]
        self.fc = nn.Sequential(
            nn.Linear(d, 64),
            nn.Linear(64, classes),
        )

        self.criterion_hard = nn.CrossEntropyLoss(reduction='none')
        self.criterion_soft = nn.MSELoss(reduction='none')

        ## init
        tot = len(list(self.parameters()))
        n = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n += 1
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                n += 2
        print(f'Init {n}, total {tot}')

    @torch.no_grad()
    def get_shape(self):
        x = torch.randn(1, *self.input_shape)
        y = self.m(x)
        return y.shape

    def forward(self, x):
        feat = self.m(x)
        feat = feat.view(feat.shape[0], -1)
        logit = self.fc(feat)
        return logit

    @torch.no_grad()
    def get_p(self, x):
        logit = self(x)
        p = F.softmax(logit, 1)
        return p

    # def get_loss(self, x, y, p, alpha=.5):
    def get_loss(self, x, y, p, alpha=.5):
        logit = self(x)

        loss_hard = self.criterion_hard(logit, y).mean()

        loss_soft = self.criterion_soft(logit, p).mean()

        loss = alpha * loss_hard + (1 - alpha) * loss_soft
        return loss


def kd():
    ckpt = 'dev-acc-41.97.pt'  # 这是我保存的teacher model的checkpoint，需要改成你自己的
    epochs = 20
    base_lr = 1e-3
    min_lr = 1e-5
    batch_size = 256
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    teacher_model = DeepCNN((3, 32, 32), 100)
    teacher_model.load_state_dict(torch.load(ckpt, 'cpu'))
    teacher_model.to(device)
    teacher_model.eval()

    model = ShallowCNN((3, 32, 32), 100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', .1, 4, False, .001, 'abs', min_lr=min_lr)

    loader, dev_loader = cifar100_loader(batch_size)

    ## validate
    teacher_dev_acc = eval(dev_loader, teacher_model, device)
    print(f'Teacher model dev acc={teacher_dev_acc * 100:.2f}%')
    teacher_model.eval()

    step = 0
    tot_steps = epochs * len(loader)
    bst = 0
    for epoch in range(1, 1 + epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                p = teacher_model(x)  # logit

            step += 1
            optimizer.zero_grad()

            start, end = .2, 0.
            k = 1 / tot_steps - 1 / tot_steps * step  # 0 -> -1
            alpha = (start - end) * k + start

            loss = model.get_loss(x, y, p, alpha)

            loss.backward()
            optimizer.step()

        acc = eval(dev_loader, model, device)
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch={epoch}, lr={lr:.2e}, alpha={alpha:.2f}'
              f', loss={loss.item():.3f}, dev_acc={acc * 100:.2f}%')
        scheduler.step(acc)

        if acc > bst:
            bst = acc
            save_name = 'bst.pt'
            torch.save(model.state_dict(), save_name)

    print(f'best dev acc={bst * 100:.2f}%')