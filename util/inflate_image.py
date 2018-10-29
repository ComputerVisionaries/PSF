import numpy as np
from puzzle.piece import Piece

def inflate_image(pieces, rows, columns):
    """Assembles image from pieces.

    Given an array of pieces and desired image dimensions, function
    assembles image by stacking pieces.

    :params pieces:  Image pieces as an array.
    :params rows:    Number of rows in resulting image.
    :params columns: Number of columns in resulting image.

    Usage::

        >>> from util.flatten_image import flatten_image
        >>> from util.inflate_image import inflate_image
        >>> pieces, rows, cols = flatten_image(...)
        >>> original_img = inflate_image(pieces, rows, cols)

    """
    vertical_stack = []
    for i in range(rows):
        horizontal_stack = []
        for j in range(columns):
            print(pieces[i * columns + j].shape)
            horizontal_stack.append(pieces[i * columns + j])
        vertical_stack.append(np.hstack(horizontal_stack))

    return np.vstack(vertical_stack).astype(np.uint8)