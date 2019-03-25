from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


# Accepts binary file path and returns an Nx4 numpy where N length 4 arrays represents [X,Y,Z, Reflectivity]
def get_points(file):
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


# Possible features, average x y or z, average reflectivity, max/mins of xyz or reflectivity.

# Accepts an Nx4 numpy array and returns a Yx256x256 grid representation of points where Y is the number of feature channels (different heights, intensity, density etc...)
def convert_points(points):

    vertical_plane_heights = [1.0, 2.0, 3.0]  # height in meters
    grid_box_size = 256
    grid = np.zeros((len(vertical_plane_heights), grid_box_size, grid_box_size))

    grid_side_len = 100.0 # meters
    grid_box_side_len = grid_side_len / grid_box_size

    return grid


class ObjectDataset(Dataset):
    def __init__(self):
        super(ObjectDataset, self).__init__()


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


def test():
    file = "../data/training/subset/000000.bin"
    x = get_points(file)
    y = convert_points(x)

    print("done")


if __name__ == '__main__':
    test()




