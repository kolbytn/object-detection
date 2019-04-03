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
    input_dir = os.path.join(training_path, 'input')  # input features must be in dir named input
    target_dir = os.path.join(training_path, 'target')  # target features must be in dir named target

    count = 0
    while max_data is None or len(training_inputs) < max_data:
        input_file = os.path.join(input_dir, str(count).zfill(6) + '.bin')  # input file must be of form 000000.bin
        target_file = os.path.join(target_dir, str(count).zfill(6) + '.txt')  # input file must be of form 000000.txt

        if os.path.isfile(input_file) and os.path.isfile(target_file):
            # Get inputs
            points = get_points(input_file)
            inputs = convert_points(points).astype('d')
            training_inputs.append(torch.from_numpy(inputs))

            # Get targets
            target = get_boxes(target_file)
            training_targets.append(torch.FloatTensor(target))
        count += 1

    training_dataset = ObjectDataset(training_inputs, training_targets)

    if testing_path is not None and testing_path != "":
        testing_inputs = []
        testing_targets = []
        input_dir = os.path.join(testing_path, 'input')  # input features must be in dir named input
        target_dir = os.path.join(testing_path, 'target')  # target features must be in dir named target

        count = 0
        while max_data is None or len(testing_inputs) < max_data:
            input_file = os.path.join(input_dir, str(count).zfill(6) + '.bin')  # input file must be of form 000000.bin
            target_file = os.path.join(target_dir, str(count).zfill(6) + '.txt')  # input file must be of form 000000.txt

            if os.path.isfile(input_file) and os.path.isfile(target_file):
                # Get inputs
                points = get_points(input_file)
                inputs = convert_points(points).astype('d')
                testing_inputs.append(torch.from_numpy(inputs))

                # Get targets
                target = get_boxes(target_file)
                testing_targets.append(torch.FloatTensor(target))
            count += 1

        testing_dataset = ObjectDataset(testing_inputs, testing_targets)
    else:
        testing_dataset = None

    return training_dataset, testing_dataset
