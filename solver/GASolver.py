#!/usr/bin/env python

from solver.AbstractSolver import AbstractSolver

"""
    Genetic Algorithm puzzle solver implementation

    Input :
        pieces : array of puzzle pieces
        width : integer number of pieces in puzzle width
        height : integer number of pieces in puzzle height

"""

class GASolver(AbstractSolver):

    def __init__(self, pieces, width, height, pSize, mRate):
        AbstractSolver.__init__(pieces, width, height)
        self._pSize = pSize
        self._mRate = mRate

        # gaps uses image_analysis to provide some time saving measures
        # TODO : see if we want/need this and if so can it be generalized to other solvers


    def run_iteration(self):

        # take current population
        # select parent pairs from population
        # for p1, p2, in pair:
        #   child = crossover(p1, p2)
        #   mutate(child)
        #   new_population.append(child)
        # cull new_population to target size

        # TODO : gaps uses a termination counter to see when to terminate early due to lack of improvement

        # population = new_population
        # dont return anything
        pass

    def get_solution(self):
        # return best solution
        pass