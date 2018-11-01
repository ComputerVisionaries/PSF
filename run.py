#!/usr/bin/env python2

from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

from util.flatten_image import flatten_image
from util.inflate_image import inflate_image
from solver.genetic_solver import GeneticSolver

from fitness.similarity_gradient import gradient_similarity
from fitness.similarity_rgb import rgb_similarity
from solver.hill_climbing_solver import cost, randomHillClimbing
from solver.simulated_annealing_solver import simulated_annealing_solver


"""
    Run puzzle solver
    Input :
        puzzle : input puzzle image
        size : pixel width/height of the pieces
        algorithm : algorithm used to solve puzzle (GA, SA, RHC)
        iters : number of iterations to run

"""


def genetic():
    # fake inputs
    imgStr = "./puzzle.jpg"
    algorithm = "GA"
    iters = 1

    img = imread(imgStr)
    pxl_widht = 16
    pxl_height = 16
    puzzle, rows, cols = flatten_image(img, pxl_widht, pxl_height, True)

    # Hyper parameters
    populationSize = 1000
    mutationRate = 0.33
    numberElites = 50

    solver = GeneticSolver(puzzle, rows, cols,
                           populationSize, mutationRate, numberElites)
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


def hill_climbing():
    full_img = plt.imread("images/baboon.jpg")
    n = 6
    puzzle = flatten_image(full_img, n, n)
    optimalCost = cost(puzzle, (n, n), rgb_similarity)
    ax = plt.subplot(1, 1, 1)
    ax.set_title("cost=%.2f" % optimalCost)
    plt.imshow(inflate_image(puzzle, n, n))
    plt.show()
    sq_iters = 2
    results = sorted([randomHillClimbing(puzzle, (n, n), rgb_similarity) for _ in range(sq_iters ** 2)], key=lambda x: x[0])
    for i in range(sq_iters ** 2):
        ax = plt.subplot(sq_iters, sq_iters, i + 1)
        ax.set_title("cost=%.2f" % results[i][0])
        plt.imshow(inflate_image(results[i][1], n, n))
    plt.tight_layout()
    plt.show()

def simulated_annealing():
    full_img = plt.imread("images/baboon.jpg")
    n = 6
    puzzle = flatten_image(full_img, n, n)
    optimalCost = cost(puzzle, (n, n), rgb_similarity)
    ax = plt.subplot(1, 1, 1)
    ax.set_title("cost=%.2f" % optimalCost)
    plt.imshow(inflate_image(puzzle, n, n))
    plt.show()
    sq_iters = 2
    results = sorted([simulated_annealing_solver(puzzle, (n, n), rgb_similarity, 10000, 10000.0) for _ in range(sq_iters ** 2)], key=lambda x: x[0])
    for i in range(sq_iters ** 2):
        ax = plt.subplot(sq_iters, sq_iters, i + 1)
        ax.set_title("cost=%.2f" % results[i][0])
        plt.imshow(inflate_image(results[i][1], n, n))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # genetic()
    # hill_climbing()
    simulated_annealing()


# flatten puzzle : convert 2d image into 1d array of pieces
# prime algorithm : feed in parameters into the specific alg (puzzle array, ?)
# for round in iters:
#   run trial of algorithm
# get best candidate
# assemble best candidate array into an image
# display solution image
