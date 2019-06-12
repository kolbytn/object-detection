import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from torch.nn.parameter import Parameter
import pdb
import matplotlib.pyplot as plt
import torchvision
import os
import gzip
import tarfile
import gc

from helpers.data import *
from helpers.functions import *

assert torch.cuda.is_available()


class DownConvLayer(nn.Module):
    def __init__(self, input_depth, output_depth, activation=nn.ReLU):
        super(DownConvLayer, self).__init__()
        self.conv_layer = nn.Conv2d(input_depth, output_depth, kernel_size=5, stride=2, padding=2)
        self.activation = activation()

    def forward(self, x):
        return self.activation(self.conv_layer(x))


class UpConvLayer(nn.Module):
    def __init__(self, input_depth, output_depth, activation=nn.ReLU):
        super(UpConvLayer, self).__init__()
        self.conv_layer = nn.ConvTranspose2d(input_depth, output_depth, kernel_size=5, stride=2,
                                             padding=2, output_padding=1)
        self.activation = activation()

    def forward(self, x):
        result = self.conv_layer(x)
        return self.activation(result)


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            DownConvLayer(3, 16),
            nn.BatchNorm2d(16),
            DownConvLayer(16, 32),
            nn.BatchNorm2d(32),
            DownConvLayer(32, 64),
            nn.BatchNorm2d(64),
            DownConvLayer(64, 128),
            nn.BatchNorm2d(128)
        )

        self.decoder = nn.Sequential(
            UpConvLayer(128, 64),
            nn.BatchNorm2d(64),
            UpConvLayer(64, 32),
            nn.BatchNorm2d(32),
            UpConvLayer(32, 16),
            nn.BatchNorm2d(16),
            UpConvLayer(16, 9, activation=nn.Sigmoid)
        )

        self.encode_fc = nn.Linear(128*16*16, 128)
        self.decode_fc = nn.Linear(128, 128*16*16)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        z = self.encode(x)
        z = self.dropout(z)
        out = self.decode(z)
        return out

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        z = self.encode_fc(x)
        return z

    def decode(self, z):
        z = self.decode_fc(z)
        z = self.decoder(z.reshape((z.shape[0], 128, 16, 16)))
        return z


def train(model, train_loader, optimizer, loss_func, e, display=False, device='cuda', vis_data=None):
    model.train()
    num = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target
        data = data
        optimizer.zero_grad()
        output = model(data)
        num += target.shape[0]
        loss = loss_func(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if display:
            display_imgs(target[0], output[0], data[0])

    if vis_data is not None and e % 10 == 0:
        model.eval()
        with torch.no_grad():
            output = model(vis_data[0].unsqueeze(0).to(device))
            display_imgs(vis_data[1], output[0], vis_data[0], e + 1000)

    return total_loss/num

viz_thresh = 0.4
def display_imgs(array0, array1, data, e):
    img2 = data.detach().cpu().clone().numpy()

    array0 = array0.detach().cpu().clone().numpy()
    # array0[array0 < viz_thresh] = 0
    # array0[array0 >= viz_thresh] = 1
    array1 = array1.detach().cpu().clone().numpy()
    # array1[array1 < viz_thresh] = 0
    # array1[array1 >= viz_thresh] = 1

    img0 = array0[0]
    for i in range(1, 9):
        img0 += (array0[i] * (i+1))

    img1 = array1[0]
    for i in range(1, 9):
        img1 += (array1[i] * (i+1))

    plt.clf()
    f, axarr = plt.subplots(2, 2)
    axarr[0,0].imshow(img0, cmap='jet',vmin=0,vmax=10)
    axarr[0,1].imshow(img1, cmap='jet',vmin=0,vmax=10)
    axarr[1,0].imshow(img2[2], cmap='jet')
    axarr[1,1].imshow(img2[0], cmap='jet')

    f.savefig('images/' + str(e) + '.png')


def test(model, test_loader, loss_func, e, device='cuda', vis_data=None):
    model.eval()
    total_loss = 0
    num = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_func(output, target).item()
            num += target.shape[0]

        if vis_data is not None:
            output = model(vis_data[0].unsqueeze(0).to(device))
            display_imgs(vis_data[1], output[0], vis_data[0], e)

        return total_loss/num


def main():
    epochs = 1001
    batch_size = 32
    test_freq = 2
    plot_freq = 50
    training_path = "data/training/full"
    testing_path = "data/testing"
    device = 'cuda'
    lr = 1e-3
    display = False

    model = AutoEncoder().to(device)

    # Load data
    data_train, data_test = load_data(training_path, testing_path=testing_path, max_data=64)
    vis_data = data_train[10]
    vis_data_test = data_test[49]
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()

    test_loss = 0
    train_losses = []
    test_losses = []
    for e in range(epochs):
        loss = train(model, loader_train, optimizer, loss_func, e, display=display, device=device, vis_data=vis_data)
        if e % test_freq == 0:
            test_loss = test(model, loader_test, loss_func, e, device=device, vis_data=vis_data_test)
        if e % plot_freq == 0:
            plot_loss(train_losses, test_losses, e)
        train_losses.append(loss)
        test_losses.append(test_loss)
        print("Epoch:", e, " Train Loss:", loss, "Test Loss:", test_loss)


if __name__ == '__main__':
    main()
