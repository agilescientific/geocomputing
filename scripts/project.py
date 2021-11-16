"""
Make a lot of Wells.
"""
import glob

import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

from well import Well


class Project:
    """
    A list of Wells.
    """
    def __init__(self, list_of_wells):
        self.list_of_wells = list_of_wells

    @classmethod
    def from_files(cls, path):

        list_of_wells = []
        for fname in glob.glob(path):
            with open(fname) as f:
                try:
                    w = Well.from_file(fname)
                    list_of_wells.append(w)
                except:
                    pass

        return cls(list_of_wells)

    def __repr__(self):
        n = len(self.list_of_wells)
        return f"Project({n} wells)"

    def __getitem__(self, idx):
        return self.list_of_wells.__getitem__(idx)

    def get_points(self, name):
        points = []
        for well in self.list_of_wells:
            top = well.get_top(name)
            if top is None:
                value = np.nan
            else:
                value = top.top
            points.append([well.x, well.y, value])
        return np.array(points)

    def map(self, name=None):
        points = self.get_points(name)
        points = points[~np.isnan(points[:, -1])]
        x, y, z = points.T

        if name is None:
            plt.scatter(x, y, c=z)
            plt.show()
            return

        extent = x_min, x_max, y_min, y_max = [x.min()-0.1,
                                               x.max()+0.1,
                                               y.min()-0.1,
                                               y.max()+0.1
                                               ]
        grid_x, grid_y = np.mgrid[x_min:x_max:0.05, y_min:y_max:0.05]
        rbfi = Rbf(x, y, z)
        di = rbfi(grid_x, grid_y)

        plt.imshow(di.T, origin="lower", extent=extent)
        plt.scatter(x, y, color='k')
        plt.title(name)
        plt.show()

        return
