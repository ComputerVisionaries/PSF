#!/usr/bin/env python

import numpy as np

class PhysicalPiece:
    """Represents single jigsaw puzzle piece.

    Each piece has identifier so it can be
    tracked across different individuals

    :param image: ndarray representing piece's RGB values
    :param index: Unique id withing piece's parent image

    Usage::

        >>> from puzzle.piece import Piece
        >>> piece = Piece(image[:28, :28, :], 42)

    """

    def __init__(self, image, index, edges):
        self.image = image[:]
        self.id = index
        self._edges = edges

        # L, B, R, T
        # 0 = edge, 1 = head, -1 = divot
        # self._sides = sides

    def getSide(self, side, numPts):

        epts = self._edges[side]
        num = len(epts) // numPts

        if num == 0:
            raise ValueError('Choose a smaller value for numPts')

        return np.array([self._edges[side][num * i] for i in range(numPts)])

    def size(self):
        """Returns piece size"""
        return self.image.shape[0]

    def shape(self):
        """Returns shape of piece's image"""
        return self.image.shape
