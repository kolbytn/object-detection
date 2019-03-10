import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


from helpers.networks import *
from helpers.data import *
from helpers.functions import *


def main():
    '''
    Get and preprocess data. Train model. Display output.
    '''
    # Set hyperparameters
    epochs = 100
    batch_size = 128
    data_path = ""
    device = "cpu"
    lr = 1e-3

    # Load data
    data_train, data_test = load_data(data_path)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)

    # Create model
    model = ObjectNetwork().to(device)
    objective = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Loop data for n epochs
    loop = tqdm(total=epochs * len(loader_train), position=0, leave=False)
    losses = []
    for epoch in epochs:
        for batch, inp, out in enumerate(loader_train):
            # Get batch input and outputs
            inp, out = inp.to(device), out.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Get model prediction
            pred = model(inp)

            # Calculate loss
            loss = objective(out, pred)

            # Comput gradients and take step
            loss.backward()
            optimizer.step()

            # Log
            losses.append(loss.item())
            loop.update(1)
            loop.set_description('epoch:{}, batch:{}, loss:{:.4f}'
                                 .format(epoch, batch, loss.item()))
        print()

    # Plot results
    plot_results(losses)


if __name__ == '__main__':
    main()
