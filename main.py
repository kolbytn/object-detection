import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from helpers.layers.modules import MultiBoxLoss
from helpers.ssd import *
from helpers.data import *
from helpers.functions import *


def main():
    '''
    Get and preprocess data. Train model. Display output.
    '''
    # Set hyperparameters
    epochs = 10000
    batch_size = 32
    training_path = "data/training/full"
    testing_path = None
    device = 'cuda'
    lr = 1e-4

    # Load data
    data_train, data_test = load_data(training_path, testing_path=testing_path, max_data=1000)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True,
                              collate_fn=detection_collate)
    loader_test = None
    if data_test:
        loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True, pin_memory=True,
                                 collate_fn=detection_collate)

    # Create model
    cfg = kitti
    model = build_ssd('train', cfg['min_dim'], cfg['num_classes']).to(device)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, device=device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Loop data for n epochs
    #loop = tqdm(total=epochs * len(loader_train), position=0, leave=False)
    losses = []
    for epoch in range(epochs):
        epoch_losses = 0.0
        for batch, data in enumerate(loader_train):
            # Get batch input and outputs
            inp, tar = data[0].to(device).float(), data[1]

            # Zero gradients
            optimizer.zero_grad()

            # Get model prediction
            pred = model(inp)

            # Calculate loss
            loss_l, loss_c = criterion(pred, tar)
            loss = loss_l + loss_c
            epoch_losses += loss.item()

            # Compute gradients and take step
            loss.backward()
            optimizer.step()

        losses.append(epoch_losses)

        print('epoch:{}, loss:{:.4f}'.format(epoch,  epoch_losses))

    # Plot results
    plot_results(losses)


if __name__ == '__main__':
    main()
