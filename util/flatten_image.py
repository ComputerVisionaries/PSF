#!/usr/bin/env python

import numpy as np
from puzzle.piece import Piece


def flatten_image(image, row_count, column_count, indexed=False):
    """Converts image into list of square pieces.

    Input image is chopped up into pieces of the specified number of rows and columns.
    Each list element is Piece that contains and id, shape, and image

    :params image: Input image.
    :params row_count: number of rows in the puzzle
    :params column_count: number of columns in the puzzle

    Usage::

        >>> from util.flatten_image import flatten_image
        >>> puzzle_array = flatten_image(image, 10, 10)

    """
    i,j,k = image.shape

    # calculate pixel height
    if i+1 % row_count:
        pixel_height = (i // row_count)
    else:
        pixel_height = (i // row_count) + 1
    
    # calculate pixel width
    if j+1 % column_count:
        pixel_width = (j // column_count)
    else:
        pixel_width = (j // column_count) + 1
    pieces = []

    top, bottom = (0, pixel_height)
    for y in range(row_count):
        left, right = (0, pixel_width)
        for x in range(column_count):
            piece = np.zeros((pixel_height, pixel_width, k))
            
            floor = min(bottom, i) - top
            wall = min(right, j) - left

            piece[:floor, :wall, :] = image[top:bottom, left:right, :]
            pieces.append(piece)

            left, right = (right, right + pixel_width)
        top, bottom = (bottom, bottom + pixel_height)

    if indexed:
        pieces = [Piece(value, index, None) for index, value in enumerate(pieces)]
    print(("=== Image flattened \n"
           "    Piece width : {0}\n"
           "    Piece height : {1}\n").format(pixel_width, pixel_height))
 
    return pieces
