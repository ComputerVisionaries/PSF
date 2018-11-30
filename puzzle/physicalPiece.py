#!/usr/bin/env python

import numpy as np


class Side:

    def __init__(self, edge, shape):
        self.edge = edge
        self.shape = shape

    # Array of RGB pixels for the given edge's side
    def get_edge(self):
        return self.edge

    # -1 divot, 1 head, 0 flat
    def get_shape(self):
        return self.shape


class PhysicalPiece:
    """Represents single jigsaw puzzle piece.

    Each piece has identifier so it can be
    tracked across different individuals

    :param image: ndarray representing piece's RGB values
    :param index: Unique id withing piece's parent image

    """

    # sides = [top_side, bottom_side, right_side, left_side] are all Side objects
    def __init__(self, image, index, sides):
        self.image = image[:]
        self.id = index
        self.sides = {
            "t": sides[0],
            "b": sides[1],
            "r": sides[2],
            "l": sides[3]
        }

    def __getitem__(self, index):
        return self.image.__getitem__(index)

    # side: "t", "r", "b", or "l"
    def get_side(self, side):
        return self.sides[side]

    def size(self):
        """Returns piece size"""
        return self.image.shape[0]

    def shape(self):
        """Returns shape of piece's image"""
        return self.image.shape
