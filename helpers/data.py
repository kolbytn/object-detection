from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import math

# Accepts binary file path and returns an Nx4 numpy where N length 4 arrays represents [X,Y,Z, Reflectivity]
def get_points(file):
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


# Points Features: x, y, z, and reflectivity
# Accepts an Nx4 numpy array and returns a Yx256x256 grid representation of points where Y is the number of feature channels (different heights, intensity, density etc...)
def convert_points(points):

    vertical_plane_heights = [1.0, 2.0, 3.0]  # height in meters
    grid_box_size = 256 # array length
    grid_side_len = 100 # meters
    grid_box_side_len_from_meter = grid_box_size / grid_side_len

    # set up 3d array
    grid = np.zeros((len(vertical_plane_heights), grid_box_size, grid_box_size), dtype=np.ndarray)
    # grid = np.array((len(vertical_plane_heights), grid_box_size, grid_box_size), dtype=np.ndarray)

    grid2D = np.zeros((grid_box_size, grid_box_size), dtype=np.ndarray)
    for x in range(grid_box_size):
        for y in range(grid_box_size):
            grid2D[x][y] = np.zeros((1, 4), dtype=np.ndarray)

    # test
    # print(points[0])
    # grid2D[0][0][0] = points[0]
    # np.append(grid2D[0][0], [points[1]], axis=0)
    # print(grid2D[0][0])

    for point in points:
        # around +- 50 * % + center
        x = math.floor(point[0] * grid_box_side_len_from_meter + 127.5)
        y = math.floor(point[1] * grid_box_side_len_from_meter + 127.5)
        # add point into 2D grid
        np.append(grid2D[0][0], [point], axis=0)

    # set up each level and calculation the targets
    for x in range(grid_box_size):
        for y in range(grid_box_size):
            points = grid2D[x][y]
            # height: level 0, attr: y
            height = 0
            for point in points:
                height += point[1]
            height /= len(points)
            grid[0][x][y] = height

            # intensity: level 1, attr: reflectivity
            intensity = 0
            for point in points:
                intensity += point[3]
            intensity /= len(points)
            grid[1][x][y] = intensity

            # density: level 2, attr: xy
            density = 0
            density = len(points)
            grid[2][x][y] = density

    return grid


class ObjectDataset(Dataset):
    def __init__(self):
        super(ObjectDataset, self).__init__()
        self.inputs


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




