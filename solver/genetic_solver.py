#!/usr/bin/env python2
from __future__ import print_function

from solver.abstract_solver import AbstractSolver

import random
import bisect
import heapq
from operator import attrgetter

from puzzle.puzzle import Puzzle
from util.piece_cache import PieceCache

SHARED_PIECE_PRIORITY = -10
BUDDY_PIECE_PRIORITY = -1

"""
    Genetic Algorithm puzzle solver implementation

    Input :
        pieces : array of puzzle pieces
        width : integer number of pieces in puzzle width
        height : integer number of pieces in puzzle height

"""


class GeneticSolver(AbstractSolver):

    TERMINATION_THRESHOLD = 10

    def __init__(self, pieces, width, height, pop_size, mut_rate, elites):
        AbstractSolver.__init__(self, pieces, width, height)
        self._pop_size = pop_size
        self._mut_rate = mut_rate
        self._elites = elites


        PieceCache.analyze_image(self._pieces)

        self._population = [Puzzle(pieces, width, height) for _ in range(pop_size)]

        self._maxima = float("-inf")
        self._termination_counter = 0

        self._done = False

    def run_iteration(self):

        if self._done:
            return

        fittest = None

        new_population = []
        elite = self._get_elite_individuals()
        new_population.extend(elite)

        selected_parents = roulette_selection(self._population)
        for mother, father in selected_parents:
            if (len(new_population) >= self._pop_size):
                break
            crossover = Crossover(mother, father)
            crossover.run()
            child = crossover.child()
            new_population.append(child)

        best = self.get_solution()

        if best.fitness <= self._maxima:
            self._termination_counter += 1
        else:
            self._termination_counter = 0
            self._maxima = best.fitness

        if self._termination_counter >= self.TERMINATION_THRESHOLD:
            print("=== Genetic Algorithm : Terminated Early")
            self._done = True

        self._population = new_population

    def _get_elite_individuals(self):
        return sorted(self._population, key=attrgetter("fitness"))[-self._elites:]

    def get_solution(self):
        # return best solution
        return max(self._population, key=attrgetter("fitness"))

# ----------------------- selection mechanic
def roulette_selection(population, elites=4):
    """Roulette wheel selection.

    Each individual is selected to reproduce, with probability directly
    proportional to its fitness score.

    :params population: Collection of the individuals for selecting.
    :params elite: Number of elite individuals passed to next generation.

    """
    fitness_values = [individual.fitness for individual in population]
    probability_intervals = [sum(fitness_values[:i + 1]) for i in range(len(fitness_values))]

    def select_individual():
        """Selects random individual from population based on fitess value"""
        random_select = random.uniform(0, probability_intervals[-1])
        selected_index = bisect.bisect_left(probability_intervals, random_select)
        return population[selected_index]

    selected = []
    for i in xrange(len(population) - elites):
        first, second = select_individual(), select_individual()
        selected.append((first, second))

    return selected

class Crossover(object):

    def __init__(self, first_parent, second_parent):
        self._parents = (first_parent, second_parent)
        self._pieces_length = len(first_parent.pieces)
        self._child_rows = first_parent.rows
        self._child_columns = first_parent.columns

        # Borders of growing kernel
        self._min_row = 0
        self._max_row = 0
        self._min_column = 0
        self._max_column = 0

        self._kernel = {}
        self._taken_positions = set()

        # Priority queue
        self._candidate_pieces = []

    def child(self):
        pieces = [None] * self._pieces_length

        for piece, (row, column) in self._kernel.iteritems():
            index = (row - self._min_row) * self._child_columns + (column - self._min_column)
            pieces[index] = self._parents[0].piece_by_id(piece)

        return Puzzle(pieces, self._child_rows, self._child_columns, shuffle=False)

    def run(self):
        self._initialize_kernel()

        while len(self._candidate_pieces) > 0:
            _, (position, piece_id), relative_piece = heapq.heappop(self._candidate_pieces)

            if position in self._taken_positions:
                continue

            # If piece is already placed, find new piece candidate and put it back to
            # priority queue
            if piece_id in self._kernel:
                self.add_piece_candidate(relative_piece[0], relative_piece[1], position)
                continue

            self._put_piece_to_kernel(piece_id, position)

    def _initialize_kernel(self):
        root_piece = self._parents[0].pieces[int(random.uniform(0, self._pieces_length))]
        self._put_piece_to_kernel(root_piece.id, (0, 0))

    def _put_piece_to_kernel(self, piece_id, position):
        self._kernel[piece_id] = position
        self._taken_positions.add(position)
        self._update_candidate_pieces(piece_id, position)

    def _update_candidate_pieces(self, piece_id, position):
        available_boundaries = self._available_boundaries(position)

        for orientation, position in available_boundaries:
            self.add_piece_candidate(piece_id, orientation, position)

    def add_piece_candidate(self, piece_id, orientation, position):
        shared_piece = self._get_shared_piece(piece_id, orientation)
        if self._is_valid_piece(shared_piece):
            self._add_shared_piece_candidate(shared_piece, position, (piece_id, orientation))
            return

        buddy_piece = self._get_buddy_piece(piece_id, orientation)
        if self._is_valid_piece(buddy_piece):
            self._add_buddy_piece_candidate(buddy_piece, position, (piece_id, orientation))
            return

        best_match_piece, priority = self._get_best_match_piece(piece_id, orientation)
        if self._is_valid_piece(best_match_piece):
            self._add_best_match_piece_candidate(best_match_piece, position, priority, (piece_id, orientation))
            return

    def _get_shared_piece(self, piece_id, orientation):
        first_parent, second_parent = self._parents
        first_parent_edge = first_parent.edge(piece_id, orientation)
        second_parent_edge = second_parent.edge(piece_id, orientation)

        if first_parent_edge == second_parent_edge:
            return first_parent_edge

    def _get_buddy_piece(self, piece_id, orientation):
        first_buddy = PieceCache.best_match(piece_id, orientation)
        second_buddy = PieceCache.best_match(first_buddy, complementary_orientation(orientation))

        if second_buddy == piece_id:
            for edge in [parent.edge(piece_id, orientation) for parent in self._parents]:
                if edge == first_buddy:
                    return edge

    def _get_best_match_piece(self, piece_id, orientation):
        for piece, dissimilarity_measure in PieceCache.best_match_table[piece_id][orientation]:
            if self._is_valid_piece(piece):
                return piece, dissimilarity_measure

    def _add_shared_piece_candidate(self, piece_id, position, relative_piece):
        piece_candidate = (SHARED_PIECE_PRIORITY, (position, piece_id), relative_piece)
        heapq.heappush(self._candidate_pieces, piece_candidate)

    def _add_buddy_piece_candidate(self, piece_id, position, relative_piece):
        piece_candidate = (BUDDY_PIECE_PRIORITY, (position, piece_id), relative_piece)
        heapq.heappush(self._candidate_pieces, piece_candidate)

    def _add_best_match_piece_candidate(self, piece_id, position, priority, relative_piece):
        piece_candidate = (priority, (position, piece_id), relative_piece)
        heapq.heappush(self._candidate_pieces, piece_candidate)

    def _available_boundaries(self, row_and_column):
        (row, column) = row_and_column
        boundaries = []

        if not self._is_kernel_full():
            positions = {
                "T": (row - 1, column),
                "R": (row, column + 1),
                "D": (row + 1, column),
                "L": (row, column - 1)
            }

            for orientation, position in positions.iteritems():
                if position not in self._taken_positions and self._is_in_range(position):
                    self._update_kernel_boundaries(position)
                    boundaries.append((orientation, position))

        return boundaries

    def _is_kernel_full(self):
        return len(self._kernel) == self._pieces_length

    def _is_in_range(self, row_and_column):
        (row, column) = row_and_column
        return self._is_row_in_range(row) and self._is_column_in_range(column)

    def _is_row_in_range(self, row):
        current_rows = abs(min(self._min_row, row)) + abs(max(self._max_row, row))
        return current_rows < self._child_rows

    def _is_column_in_range(self, column):
        current_columns = abs(min(self._min_column, column)) + abs(max(self._max_column, column))
        return current_columns < self._child_columns

    def _update_kernel_boundaries(self, row_and_column):
        (row, column) = row_and_column
        self._min_row = min(self._min_row, row)
        self._max_row = max(self._max_row, row)
        self._min_column = min(self._min_column, column)
        self._max_column = max(self._max_column, column)

    def _is_valid_piece(self, piece_id):
        return piece_id is not None and piece_id not in self._kernel


def complementary_orientation(orientation):
    return {
        "T": "D",
        "R": "L",
        "D": "T",
        "L": "R"
    }.get(orientation, None)
