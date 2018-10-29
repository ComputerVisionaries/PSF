#!/usr/bin/env python

import numpy as np
from puzzle.piece import Piece


def flatten_image(image, pixel_width, pixel_height, indexed=False):
    """Converts image into list of square pieces.

    Input image is divided into square pieces of specified size and than
    flattened into list. Each list element is PIECE_SIZE x PIECE_SIZE x 3

    :params image: Input image.
    :params pixel_width: num of pixels across a piece is
    :params pixel_height: num of pixels high a piece is
    :params indexed: If True list of Pieces with IDs will be returned, otherwise just plain list of ndarray pieces

    Usage::

        >>> from util.flatten_image import flatten_image
        >>> puzzle_array = flatten_image(image, 32)

    """
    rows = image.shape[0] // pixel_height
    columns = image.shape[1] // pixel_width
    pieces = []

    # Crop pieces from original image
    for y in range(rows):
        for x in range(columns):
            left, top, w, h = x * pixel_width, y * pixel_height, (x + 1) * pixel_width, (y + 1) * pixel_height
            piece = np.empty((pixel_height, pixel_width, image.shape[2]))
            piece[:] = image[top:h, left:w, :]
            pieces.append(piece)

    if indexed:
        pieces = [Piece(value, index) for index, value in enumerate(pieces)]

    return pieces, rows, columns
