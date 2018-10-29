#!/usr/bin/env python

"""
    Abstract interface for the GA, SA, and RHC solvers to implement

    Input :
        pieces : array of puzzle pieces
        width : integer number of pieces in puzzle width
        height : integer number of pieces in puzzle height

"""

class AbstractSolver:

    def __init__(self, pieces, width, height):
        self._pieces = pieces
        self._width = width
        self.height = height

    def run_iteration(self):
        pass

    def get_solution(self):
        pass
