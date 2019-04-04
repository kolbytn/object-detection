import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from helpers.data import *


def plot_results(data):
    plt.plot(data)
    plt.show()


def load_data(training_path, testing_path=None, max_data=None):
    '''
    input features must be in dir named input
    target features must be in dir named target
    input and targets will be saved to a dir named saved
    input file must be of form 000000.bin
    input file must be of form 000000.txt
    '''

    # Load training data
    training_inputs, training_targets = process_data(training_path, max_data)

    training_dataset = ObjectDataset(training_inputs, training_targets)

    # If testing, load testing data
    if testing_path is not None and testing_path != "":
        testing_inputs, testing_targets = process_data(testing_path, max_data)

        testing_dataset = ObjectDataset(testing_inputs, testing_targets)
    else:
        testing_dataset = None

    return training_dataset, testing_dataset


def process_data(path, max_data):
    saved_input_dir = 'saved/input.npy'
    saved_target_dir = 'saved/target.npy'

    # Load training data
    inputs = []
    targets = []

    # Attempt to load preprocessed data
    input_file = os.path.join(path, saved_input_dir)
    target_file = os.path.join(path, saved_target_dir)
    if os.path.isfile(input_file) and os.path.isfile(target_file):
        inp = np.load(input_file)
        for i in range(inp.shape[0]):
            inputs.append(torch.from_numpy(inp[i]))
        tar = np.load(target_file)
        for i in range(tar.shape[0]):
            targets.append(torch.from_numpy(tar[i]))

    # Load and process data
    else:
        input_dir = os.path.join(path, 'input')
        target_dir = os.path.join(path, 'target')
        count = 0
        while max_data is None or len(inputs) < max_data:
            input_file = os.path.join(input_dir, str(count).zfill(6) + '.bin')
            target_file = os.path.join(target_dir, str(count).zfill(6) + '.txt')

            if os.path.isfile(input_file) and os.path.isfile(target_file):
                # Get inputs
                points = get_points(input_file)
                inp = convert_points(points).astype('d')
                inputs.append(torch.from_numpy(inp))

                # Get targets
                tar = get_boxes(target_file)
                targets.append(torch.FloatTensor(tar))
            else:
                break
            count += 1
            print(count)
        np.save(os.path.join(path, saved_input_dir), [x.numpy() for x in inputs])
        np.save(os.path.join(path, saved_target_dir), [x.numpy() for x in targets])
        
    return inputs, targets


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

