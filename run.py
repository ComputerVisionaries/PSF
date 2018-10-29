#!/usr/bin/env python2

from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

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
    imgStr = "./puzzle.jpg"
    algorithm = "GA"
    iters = 80

    img = imread(imgStr)
    pxl_widht = 16
    pxl_height = 16
    puzzle, rows, cols = flatten_image(img, pxl_widht, pxl_height, True)

    # Hyper parameters
    populationSize = 1000
    mutationRate = 0.33
    numberElites = 50

    solver = GeneticSolver(puzzle, rows, cols, populationSize, mutationRate, numberElites)
    scores = []

    for r in range(iters):
        print("=== Solver : {}".format(r))
        solver.run_iteration()
        solution = solver.get_solution()
        scores.append(solution.fitness)

    solved_puzzle = solver.get_solution()
    imgOut = solved_puzzle.to_image()
    imsave("solved_puzzle.jpg", imgOut)

    plt.plot(scores)
    plt.show()
    

# flatten puzzle : convert 2d image into 1d array of pieces
# prime algorithm : feed in parameters into the specific alg (puzzle array, ?)
# for round in iters:
#   run trial of algorithm
# get best candidate
# assemble best candidate array into an image
# display solution image
