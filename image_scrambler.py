import numpy as np
from scipy.misc import imread, imsave

def image_scramble(img, x_tiles, y_tiles):
    imrows, imcols = img.shape[:-1]
    if imrows % y_tiles != 0:
        raise ValueError('Cannot Divide the Puzzle Vertically')
    if imcols % x_tiles != 0:
        raise ValueError('Cannot Divide the Puzzle Horizontally')
    size_tile_x = imcols / x_tiles
    size_tile_y = imrows / y_tiles
    pieces = np.zeros((x_tiles * y_tiles, size_tile_y, size_tile_x, 3))
    num_tiles = x_tiles * y_tiles
    index = 0

    # Loop through the original image to get the tiles we need into a flat-tile array
    for y in range(y_tiles):
        for x in range(x_tiles):
            start_x = x * size_tile_x
            start_y = y * size_tile_y
            tile = img[start_y:(start_y + size_tile_y), start_x:(start_x + size_tile_x)]
            pieces[index] = tile
            index += 1

    # Decide on a new permutation for the pieces
    new_permutation = np.random.permutation(num_tiles)
    new_im = np.zeros(img.shape)

    # Place the tiles in the new permutation
    for i in range(new_permutation.size):
        y = (i * size_tile_x) / imcols
        x = ((i * size_tile_x) % imcols) / size_tile_x
        y *= size_tile_y
        x *= size_tile_x
        new_im[y:(y + size_tile_y), x:(x + size_tile_x)] = pieces[new_permutation[i]]

    return new_im


if __name__=='__main__':
    img = imread('creatures_pic.jpg')
    scrambled = image_scramble(img, 40, 40)
    imsave('creatures_scrambled.png', scrambled)





