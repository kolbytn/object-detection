import pykitti
from pykitti import utils
import os
import numpy as np
import glob


def get_points(file):
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))



if __name__ == '__main__':
    file = "points/onefile/000002.bin"
    get_points(file)


    # 3x256x256

    print("done")








