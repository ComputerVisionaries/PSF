#!/usr/bin/env python2

"""Create puzzle from input image.

This file loads image and creates puzzle by dividing image
into square pieces and then shuffles pieces to produce random puzzle.

"""

# SYSTEM IMPORTS
import argparse
import os.path
import numpy as np
from scipy.misc import imread, imsave

# PROJECT IMPORTS
from util.flatten_image import flatten_image
from util.inflate_image import inflate_image

DEFAULT_PUZZLE_WIDTH = 10
DEFAULT_PUZZLE_HEIGHT = 10
DEFAULT_OUTPUT_DEST = "./puzzle.jpg"

COLOR_STRING = {
    "ERROR": "\033[31m[ERROR]\033[0m {0}",
    "SUCCESS": "\033[32m[SUCCESS]\033[0m {0}"
}

def print_messages(messages, level="SUCCESS"):
    """Prints given messages as colored strings"""
    print
    for message in messages:
        print COLOR_STRING[level].format(message)

def parse_arguments():
    """Parses input arguments required to create puzzle"""
    description = ("Create puzzle pieces from input image by random shuffling.\n"
                   "Maximum possible rectangle is cropped from original image.")


    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("source",
                        type=str,
                        help="Path to the input file.")
    
    parser.add_argument("--destination", "-d",
                        type=str,
                        default=DEFAULT_OUTPUT_DEST,
                        help="Path to the output file.")

    parser.add_argument("--rows", "-r",
                        type=int,
                        default=DEFAULT_PUZZLE_HEIGHT,
                        help="Number of rows in the puzzle")

    parser.add_argument("--cols", "-c",
                        type=int,
                        default=DEFAULT_PUZZLE_WIDTH,
                        help="Number of columns in the puzzle")

    return parser.parse_args()

def validate_arguments(args):
    """Validates input arguments required to create puzzle"""
    errors = []

    if not os.path.isfile(args.source):
        errors.append("Image does not exist.")

    if len(errors) > 0:
        print_messages(errors, level="ERROR")
        exit()


def image_scramble(image_path, output_path, rows, columns):

    image = imread(image_path)
    pieces = flatten_image(image, rows, columns)
    np.random.shuffle(pieces)
    puzzle = inflate_image(pieces, rows, columns)
    imsave(output_path, puzzle)
    print_messages([("Puzzle created: \n"
                     "Rows: {0}\n"
                     "Columns: {1}\n\n"
                     "Puzzle can be found at: {2}").format(rows, columns, output_path)])


if __name__=='__main__':
    ARGS = parse_arguments()
    validate_arguments(ARGS)
    image_scramble(ARGS.source, ARGS.destination, ARGS.rows, ARGS.cols)





