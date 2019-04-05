import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import mayavi.mlab

from helpers.data import *


def plot_loss(train_losses, test_losses):
    plt.clf()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.legend(loc=1)
    plt.title('BCE Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig('images/loss_' + str(e) + '.png')


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
            inputs.append(torch.FloatTensor(inp[i]))
        tar = np.load(target_file)
        for i in range(tar.shape[0]):
            targets.append(torch.FloatTensor(tar[i]))

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
                inputs.append(torch.FloatTensor(inp).unsqueeze(0))

                # Get targets
                tar = get_boxes(target_file)
                targets.append(torch.FloatTensor(tar).unsqueeze(0))
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


def display(image, labels, y):
    plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1

    plt.show()

def visualize(file):
    # Plot using mayavi -Much faster and smoother than matplotlib

    scan = np.fromfile(file, dtype=np.float32)
    velo = scan.reshape((-1, 4))

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mayavi.mlab.points3d(
        velo[:, 0],  # x
        velo[:, 1],  # y
        velo[:, 2],  # z
        velo[:, 2],  # Height data used for shading
        mode="point",  # How to render each point {'point', 'sphere' , 'cube' }
        colormap='spectral',  # 'bone', 'copper',
        # color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
        scale_factor=100,  # scale of the points
        line_width=10,  # Scale of the line, if any
        figure=fig,
    )
    # velo[:, 3], # reflectance values
    mayavi.mlab.show()


if __name__ == '__main__':
    visualize("../data/training/full/input/000000.bin")