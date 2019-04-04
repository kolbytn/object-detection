from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import math

'''
Inputs: list of points, x, y, z, and reflectivity
Targets: list of
    Object: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
    Truncated: Whether object leaves boundaries (0-1)
    Visibility: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
    Angle: observation angle of object, ranging [-pi..pi]
    BBox-Left
    BBox-Top
    BBox-Right
    BBox-Bottom
    Height
    Width
    Length
    X-Location
    Y-Location
    Z-Location
    Y-Rotation [-pi..pi]
'''


# TODO These need to be set to correct values
kitti = {
    'num_classes': 9,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'KITTI'
}


object_classes = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
}


# Accepts binary file path and returns an Nx4 numpy where N length 4 arrays represents [X,Y,Z, Reflectivity]
def get_points(file):
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


# Points Features: x, y, z, and reflectivity
# Accepts an Nx4 numpy array and returns a Yx256x256 grid representation of points where Y is the number of feature channels (different heights, intensity, density etc...)
def convert_points(points):

    vertical_plane_heights = [1.0, 2.0, 3.0]  # height in meters
    grid_box_size = 300  # array length
    grid_side_len = 100  # meters

    # set up 3d array
    grid = np.zeros((len(vertical_plane_heights), grid_box_size, grid_box_size), dtype=np.ndarray)
    # grid = np.array((len(vertical_plane_heights), grid_box_size, grid_box_size), dtype=np.ndarray)

    grid2D = []
    for x in range(grid_box_size):
        grid2D.append([])
        for y in range(grid_box_size):
            grid2D[x].append([])

    for point in points:
        # ignore when value greater than +-50
        if (abs(point[0]) > 50 or abs(point[1]) > 50): continue
        # value boundary: +- 50
        # (point(x or y) +50)/100 * (256 - 1)
        tempX = math.floor((point[0] + grid_side_len/2) / grid_side_len * (grid_box_size - 1))
        tempY = math.floor((point[2] + grid_side_len/2) / grid_side_len * (grid_box_size - 1))
        x = min(tempX, 255)
        y = min(tempY, 255)
        # add point into 2D grid
        grid2D[x][y].append(point)

    # set up each level and calculation the targets
    for x in range(grid_box_size):
        for y in range(grid_box_size):
            points = grid2D[x][y]
            # height: level 1, attr: z
            height = 0
            for point in points:
                height += point[2]
            if(len(points) != 0):
                height /= len(points)
            grid[0][x][y] = height

            # intensity: level 2, attr: reflectivity
            intensity = 0
            for point in points:
                intensity += point[3]
            if (len(points) != 0):
                intensity /= len(points)
            grid[1][x][y] = intensity

            # density: level 3, attr: number points in square
            density = 0
            density = len(points)
            grid[2][x][y] = density

    return grid


def get_boxes(file_path):
    # Get objects
    with open(file_path, mode='r') as file:
        objects_str = file.readlines()

        objects = []
        for object_str in objects_str:
            items = object_str.split(' ')
            items[0] = object_classes[items[0]]
            if items[0] == 8:
                continue
            items = [float(x) for x in items]
            obj = [items[11], items[12], items[11] + items[9], items[12] + items[10], items[0] - 1]
            objects.append(obj)

    # Returns [[xmin, ymin, xmax, ymax, label_idx], ... ]
    return objects


class ObjectDataset(Dataset):
    def __init__(self, inputs, targets):
        super(ObjectDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


def test():
    file = "../data/training/subset/input/000000.bin"
    x = get_points(file)
    y = convert_points(x)

    print("done")


if __name__ == '__main__':
    test()




