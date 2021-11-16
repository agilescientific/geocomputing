#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tops parser.

"""
import argparse
import glob
from dataclasses import dataclass
import logging

import matplotlib.pyplot as plt

from utils import get_data

logging.basicConfig(level=logging.INFO)


class TopsException(Exception):
    pass


@dataclass
class Top:
    well: str
    x: float
    y: float
    top: float
    base: float

    @property
    def thickness(self):
        return self.base - self.top


class Formation:
    def __init__(self, name, list_of_tops):
        self.name = name
        self.tops = list_of_tops
        return

    @classmethod
    def from_tops(cls, name, path):
        list_of_tops = []
        for fname in glob.glob(path):
            try:
                uwi, lon, lat, top, base = get_data(name, fname)
                this_top = Top(uwi, lon, lat, top, base)
                list_of_tops.append(this_top)
                logging.info(f"Loaded a file {fname}")
            except AttributeError:
                logging.warning(f"There was a problem reading {fname}")

        return cls(name, list_of_tops)

    @property
    def thickness(self):
        return [t.thickness for t in self.tops]

    @property
    def x(self):
        return [t.x for t in self.tops]

    @property
    def y(self):
        return [t.y for t in self.tops]

    @property
    def z(self):
        return [t.top for t in self.tops]

    def map(self):
        plt.scatter(self.x, self.y, c=self.z)
        plt.colorbar()
        plt.show()
        pass


def make_a_map(name, path, outfile):
    # Make a map from files in path
    formation = Formation.from_tops(name, path)
    formation.map()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('-i', '--infile')
    parser.add_argument('-o', '--outfile')
    args = parser.parse_args()

    make_a_map(args.name, args.infile, args.outfile)


    # parser = argparse.ArgumentParser(description='Make a map from tops files.')
    # parser.add_argument('path',
    #                     metavar='Path to tops files',
    #                     type=str,
    #                     nargs='?',
    #                     default='./*.tops',
    #                     help='The path to one or more .tops files. Uses Unix-style pathname expansion. Omit to find all .tops files in current directory.')
    # parser.add_argument('-i', '--infile',
    #                     metavar='input file',
    #                     type=str,
    #                     nargs='?',
    #                     help='The path to an input file.)
    # parser.add_argument('-o', '--outfile',
    #                     metavar='output file',
    #                     type=str,
    #                     nargs='?',
    #                     default='./map.png',
    #                     help='The path to an output file. Default: map.png in current directory.')
    # args = parser.parse_args()
