#!/usr/bin/env python

class Piece(object):
    """Represents single jigsaw puzzle piece.

    Each piece has identifier so it can be
    tracked across different individuals

    :param image: ndarray representing piece's RGB values
    :param index: Unique id withing piece's parent image

    Usage::

        >>> from puzzle.piece import Piece
        >>> piece = Piece(image[:28, :28, :], 42)

    """

    def __init__(self, image, index, sides):
        self.image = image[:]
        self.id = index

        # T = top, B = bottom, L = left, R = right
        # 0 = edge, 1 = head, -1 = divot
        self._sides = sides

    def __getitem__(self, index):
        return self.image.__getitem__(index)

    def size(self):
        """Returns piece size"""
        return self.image.shape[0]

    def shape(self):
        """Returns shape of piece's image"""
        return self.image.shape
