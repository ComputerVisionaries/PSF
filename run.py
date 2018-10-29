#!/usr/bin/env python

from scipy.misc import imread, imsave

from util.flatten_image import flatten_image
from util.inflate_image import inflate_image
from solver.genetic_solver import GeneticSolver

"""
    Run puzzle solver
    Input :
        puzzle : input puzzle image
        size : pixel width/height of the pieces
        algorithm : algorithm used to solve puzzle (GA, SA, RHC)
        iters : number of iterations to run

"""

if __name__ == "__main__":
    # fake inputs
    imgStr = "./puzzle.png"
    size = 64  # TODO eleminate size param ??
    algorithm = "GA"
    iters = 20

    img = imread(imgStr)
    puzzle, rows, cols = flatten_image(img, 64, True)

    # Hyper parameters
    populationSize = 100
    mutationRate = 0.1
    numberElites = 2

    solver = GeneticSolver(puzzle, rows, cols, populationSize, mutationRate, numberElites)

    for _ in range(iters):
        solver.run_iteration()

    solved_puzzle = solver.get_solution()
    imgOut = solved_puzzle.to_image()
    imsave("solved_puzzle.jpg", imgOut)

# flatten puzzle : convert 2d image into 1d array of pieces
# prime algorithm : feed in parameters into the specific alg (puzzle array, ?)
# for round in iters:
#   run trial of algorithm
# get best candidate
# assemble best candidate array into an image
# display solution image
