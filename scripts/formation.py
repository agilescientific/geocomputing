# -*- coding:utf-8 -*-
"""
Classes for handling geological formations.
"""
from dataclasses import dataclass
import glob
import matplotlib.pyplot as plt
import argparse
import logging

import utils

logging.basicConfig(level=logging.WARNING)


@dataclass
class Top:
    """
    One top, or point in our pointset.
    """
    well: str
    x: float
    y: float
    top: float
    base: float

    @property
    def thickness(self):
        return self.base - self.top


class Formation:
    """
    One formation.
    """
    def __init__(self, name, list_of_tops):
        self.name = name
        self.list_of_tops = list_of_tops
        return

    @classmethod
    def from_files(cls, name, path):
        """
        Read a lot of files and make a Formation instance.
        """
        list_of_tops = []
        for fname in glob.glob(path):
            try:
                well, x, y, top, base = utils.file_reader(name, fname)
                this_top = Top(well, x, y, top, base)
                list_of_tops.append(this_top)
                logging.info(f"{name}: Loaded {fname}")
            except:
                logging.warning(f"{name}: Skipped {fname}")
                pass
        return cls(name, list_of_tops)

    def __eq__(self, other):
        return self.name == other.name

    @classmethod
    def from_csv(cls):
        raise(NotImplementedError)

    @property
    def x(self):
        return [t.x for t in self.list_of_tops]

    @property
    def y(self):
        return [t.y for t in self.list_of_tops]

    @property
    def z(self):
        return [t.top for t in self.list_of_tops]

    @property
    def thickness(self):
        return [t.thickness for t in self.list_of_tops]

    @property
    def base():
        return [t.base for t in self.list_of_tops]

    @property
    def gradient():
        return utils.compute_gradient(self.x, self.y, self.z)

    # More properties

    def make_map(self, attr, outfile=None):
        plt.scatter(self.x, self.y, c=getattr(self, attr))
        if outfile is not None:
            plt.savefig(outfile)
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make a thickness map from a pointset")
    parser.add_argument('name',
                        type=str,
                        # default='TD',
                        help="The Posix path to one or more tops files."
                        )
    parser.add_argument('path')
    args = parser.parse_args()

    f = Formation.from_files(args.name, args.path)
    f.make_map('thickness')
