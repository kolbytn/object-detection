import matplotlib.pyplot as plt
import torch
import os

from helpers.data import *


def plot_results(data):
    plt.plot(data)
    plt.show()


def load_data(training_path, testing_path=None, max_data=None):
    training_inputs = []
    training_targets = []
    directory = os.fsencode(training_path)
    for file in os.listdir(directory):
        if max_data and len(training_inputs) >= max_data:
            break
        filename = os.fsdecode(file)
        if filename.endswith(".bin"):
            points = get_points(os.path.join(training_path, filename))
            inputs = convert_points(points).astype('d')
            training_inputs.append(torch.from_numpy(inputs))
            training_targets.append(torch.zeros(1))
    training_dataset = ObjectDataset(training_inputs, training_targets)

    if testing_path is not None and testing_path != "":
        testing_inputs = []
        testing_targets = []
        directory = os.fsencode(testing_path)
        for file in os.listdir(directory):
            if max_data and len(training_inputs) >= max_data:
                break
            filename = os.fsdecode(file)
            if filename.endswith(".bin"):
                points = get_points(os.path.join(testing_path, filename))
                inputs = convert_points(points).astype('d')
                testing_inputs.append(torch.from_numpy(inputs))
                testing_targets.append(np.zeros(1))
        testing_dataset = ObjectDataset(testing_inputs, testing_targets)
    else:
        testing_dataset = None

    return training_dataset, testing_dataset
