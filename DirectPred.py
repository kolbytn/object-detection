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
from helpers.ssd import *
from helpers.data import *
from helpers.functions import *
from helpers.cancer_model import *
import torchvision
import os
import gzip
import tarfile
import gc

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
            DownConvLayer(16, 32),
            DownConvLayer(32, 64),
            DownConvLayer(64, 128))

        self.decoder = nn.Sequential(
            UpConvLayer(128, 64),
            UpConvLayer(64, 32),
            UpConvLayer(32, 16),
            UpConvLayer(16, 9, activation=nn.Sigmoid)
        )

        self.encode_fc = nn.Linear(128*16*16, 128)
        self.decode_fc = nn.Linear(128, 128*16*16)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        z = self.encode_fc(x)
        return z

    def decode(self, z):
        z = self.decode_fc(z)
        z = self.decoder(z.reshape((z.shape[0], 128, 16, 16)))
        return z


def train(model, train_loader, optimizer, loss_func, display=False):
    model.train()
    num = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cuda"), target.to("cuda")
        target = target
        data = data
        optimizer.zero_grad()
        output = model(data)
        num += target.shape[0]
        loss = loss_func(output, target)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if display:
            display_imgs(target[0], output[0], data[0])

    return total_loss.item()/num

viz_thresh = 0.4
def display_imgs(array0, array1, data):
    img2 = data.detach().cpu().numpy()

    array0 = array0.detach().cpu().numpy()
    array0[array0 < viz_thresh] = 0
    array0[array0 >= viz_thresh] = 1
    array1 = array1.detach().cpu().numpy()
    array1[array1 < viz_thresh] = 0
    array1[array1 >= viz_thresh] = 1

    img0 = array0[0]
    for i in range(1, 9):
        img0 += (array0[i] * (i+1))

    img1 = array1[0]
    for i in range(1, 9):
        img1 += (array1[i] * (i+1))

    f, axarr = plt.subplots(2, 2)
    axarr[0,0].imshow(img0, cmap='jet',vmin=0,vmax=10)
    axarr[0,1].imshow(img1, cmap='jet',vmin=0,vmax=10)
    axarr[1,0].imshow(img2[2], cmap='jet')
    axarr[1,1].imshow(img2[0], cmap='jet')

    plt.show()


def test(model, test_loader, loss_func, display=False):
    model.eval()
    total_loss = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            total_loss += loss_func(output, data)
            num += output.shape[0]

        return total_loss.item()/num


def main():
    epochs = 1000
    batch_size = 8
    training_path = "data/training/full"
    testing_path = None
    device = 'cuda'
    lr = 5e-4
    display = False

    model = AutoEncoder().cuda()

    # Load data
    data_train, data_test = load_data(training_path, testing_path=testing_path, max_data=64)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()

    for e in range(epochs):
        if e % 50 == 0 and e >= 0:
            display= True
        loss = train(model, loader_train, optimizer, loss_func, display=display)
        print("Epoch: ", e, " Train Loss: ", loss)


if __name__ == '__main__':
    main()
