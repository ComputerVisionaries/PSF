#!/usr/bin/env python2

# SYSTEM IMPORTS
import argparse
from scipy.misc import imread, imshow, imsave
import matplotlib.pyplot as plt
from time import time

# MODULE IMPORTS
from util.flatten_image import flatten_image
from util.inflate_image import inflate_image
from util.progress_bar import print_progress
from solver.genetic_solver import GeneticSolver
#from solver.hill_climbing_solver import


"""PSF : Puzzle Solver Framework

This module deals with the creation and solution of puzzles.
PSF is highly configurable hosting various algorithms for puzzle solving

Algorithms so far:
    - Genetic
    - Random Restart Hill Climbing
    - Simulated Annealing

"""

# DEFAULT SOLVER PARAMETERS
DEFAULT_INPUT_PUZZLE = "./puzzle.jpg"
DEFAULT_OUTPUT_FOLDER = "output/"

DEFAULT_PUZZLE_WIDTH = 10
DEFAULT_PUZZLE_HEIGHT = 10

# DEFAULT GENETIC ALGORITHM HYPER-PARAMETERS
POPULATION = 200
ELITE_POPULATION = 4
MUTATION_RATE = 0.1


def parse_arguments():
    """Parses input arguments required to solve puzzle"""
    parser = argparse.ArgumentParser(description="A versatile solver for jigsaw puzzles")
    
    parser.add_argument("--image", "-i",
                        type=str,
                        default=DEFAULT_INPUT_PUZZLE,
                        help="Input puzzle image")

    parser.add_argument("--output", "-o",
                        type=str,
                        default=DEFAULT_OUTPUT_FOLDER,
                        help="Output folder location")
    
    parser.add_argument("--rows", "-r",
                        type=int,
                        default=DEFAULT_PUZZLE_HEIGHT,
                        help="Number of rows in the puzzle")

    parser.add_argument("--cols", "-c",
                        type=int,
                        default=DEFAULT_PUZZLE_WIDTH,
                        help="Number of columns in the puzzle")

    parser.add_argument("--algorithm", "-a",
                        type=str,
                        default="GA",
                        help="Algorithm to run (GA, RHC, or SA)")

    parser.add_argument("--iterations", "-iter",
                        type=int,
                        help="Num of iterations to run the algorithm for")

    # GENETIC ALGORITHM PARAMETERS    

    parser.add_argument("--population", "-p",
                        type=int,
                        default=POPULATION,
                        help="Size of population.")
    
    parser.add_argument("--elite", "-e",
                        type=int,
                        default=ELITE_POPULATION,
                        help="Size of elite population")

    parser.add_argument("--mutation", "-m",
                        type=float,
                        default=MUTATION_RATE,
                        help="% chance a mutation will occure")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    
    # base params
    image_path = args.image
    output_folder = args.output

    rows = args.rows
    cols = args.cols

    algorithm = args.algorithm
    iterations = args.iterations
    

    image = imread(args.image)
    puzzle = flatten_image(image, rows, cols, True)

    if (args.algorithm is "GA"):
        print "\n=== Genetic algorithm selected"
        print "=== Population:  {}".format(args.population)
        print "=== Elites :  {}".format(args.elite)
        print "=== Mutation Rate :  {}".format(args.mutation)
 
        solver = GeneticSolver(puzzle, rows, cols,
                               args.population, args.mutation, args.elite)

    if (args.algorithm is "RHC"):
        pass

    start = time()
    for iteration in range(iterations):
        print_progress(iteration, iterations-1, prefix="=== Solving puzzle: ")
        solver.run_iteration()

        if args.algorithm is "GA" and solver.done:
            print_progress(iterations, iterations, prefix="=== Solving puzzle: ")
            print "=== solver terminated @ round : {}".format(iteration)
            break
            

    end = time()
    print "\n=== Done in {0:.3f} s".format(end - start)

    solved_puzzle = solver.get_solution()
    imgOut = solved_puzzle.to_image()
    imsave(args.output + "solved_puzzle.jpg", imgOut)

