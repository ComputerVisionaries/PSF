import numpy as np
from itertools import combinations
from fitness.similarity_gradient import gradient_similarity
from fitness.RelativePosition import RelativePosition
import matplotlib.pyplot as plt

# Copied from util/flatten_image.py until we fix module structure
def flatten_image(image, piece_size):
    rows, columns = image.shape[0] // piece_size, image.shape[1] // piece_size
    pieces = []

    # Crop pieces from original image
    for y in range(rows):
        for x in range(columns):
            left, top, w, h = x * piece_size, y * piece_size, (x + 1) * piece_size, (y + 1) * piece_size
            piece = np.empty((piece_size, piece_size, image.shape[2]))
            piece[:piece_size, :piece_size, :] = image[top:h, left:w, :]
            pieces.append(piece)

    return pieces, (rows, columns)

# Copied from util/inflate_image.py until we fix module structure
def inflate_image(pieces, rows, columns):
    vertical_stack = []
    for i in range(rows):
        horizontal_stack = []
        for j in range(columns):
            horizontal_stack.append(pieces[i * columns + j])
        vertical_stack.append(np.hstack(horizontal_stack))

    return np.vstack(vertical_stack).astype(np.uint8)


# For small puzzles, normal discrete random hill climbing will suffice.
# For # of pieces n, number of neighboring configs is (n*(n-1)/2), O(n^2)
# Assumes 'pieces' is 1d array representing a configuration of pieces.
# Written as a generator so that large puzzles consume less memory.
def neighbors(pieces):
	inds = range(len(pieces))
	for i, j in combinations(inds, 2):
		copy = pieces.copy()
		copy[i], copy[j] = copy[j].copy(), copy[i].copy()
		yield copy

# Total cost of a puzzle configuration
def cost(pieces, dimensions, similarityFunction):
	w, h = dimensions
	c = 0
	for i in range(w):
		for j in range(h):
			p1 = pieces[i + j * w]
			for x, y, rel in [
				# Right piece
				(i + 1, j, RelativePosition.LEFT_RIGHT),
				# Below piece
				(i, j + 1, RelativePosition.ABOVE_BELOW)]:
				if 0 <= x < w and 0 <= y < h:
					p2 = pieces[x + y * w]
					c += similarityFunction(p1, p2, pos=rel)
	return c


def hillClimbingWithRestarts(pieces, dimensions, similarityFunction, restarts):
	pieces = np.random.permutation(pieces)
	best_cost, best_config = cost(pieces, dimensions, similarityFunction), pieces
	for _ in range(restarts + 1):
		cst, cfg = randomHillClimbing(pieces, dimensions, similarityFunction)
		best_cost, best_config = min((best_cost, best_config), (cst, cfg), key=lambda x: x[0])
	return (best_cost, best_config)

def randomHillClimbing(pieces, dimensions, similarityFunction):
	currConfig = np.random.permutation(pieces)
	currCost = cost(currConfig, dimensions, similarityFunction)
	while True:
		old = currCost
		for cfg in neighbors(currConfig):
			cst = cost(cfg, dimensions, similarityFunction)
			currCost, currConfig = min((currCost, currConfig), (cst, cfg), key=lambda x: x[0])
		if old == currCost:
			# We've reached a local maximum; no swap will reduce cost
			break
	print("Local max at cost=%.2f" % currCost)
	return (currCost, currConfig)

if __name__ == "__main__":
	full_img = plt.imread("../images/baboon.jpg")
	n = 4
	puzzle, _ = flatten_image(full_img, int(512 / n))
	optimalCost = cost(puzzle, (n, n), rgb_similarity)
	ax = plt.subplot(1, 1, 1)
	ax.set_title("cost=%.2f" % optimalCost)
	plt.imshow(inflate_image(puzzle, n, n))
	plt.show()
	# iterations = 10
	# bestcost, best = hillClimbingWithRestarts(puzzle, (n, n), rgb_similarity, iterations)
	sq_iters = 2
	results = sorted([randomHillClimbing(puzzle, (n, n), gradient_similarity) for _ in range(sq_iters ** 2)], key=lambda x: x[0])
	for i in range(sq_iters ** 2):
		ax = plt.subplot(sq_iters, sq_iters, i + 1)
		ax.set_title("cost=%.2f" % results[i][0])
		plt.imshow(inflate_image(results[i][1], n, n))
	plt.tight_layout()
	plt.show()

	'''
	# Below is a simplified toy problem for testing the hill climbing solver in isolation.
	def toySimilarity(pieceOne, pieceTwo, pos):
		return abs(pieceOne - pieceTwo)

	def printPuzzle(p, dimensions):
		print("cost", cost(p, dimensions, toySimilarity))
		for y in range(dimensions[0]):
			i = y * dimensions[1]
			print(p[i:i+dimensions[1]])

	n = 5
	puzzle = np.random.permutation(list(range(n ** 2)))
	dim = (n, n)

	bestcost, best = randomHillClimbing(puzzle, dim, toySimilarity)
	print("Single iteration:")
	printPuzzle(best, dim)
	iterations = 10
	bestcost, best = hillClimbingWithRestarts(puzzle, dim, toySimilarity, iterations)
	print(str(iterations) + " Iterations:")
	printPuzzle(best, dim)
	'''
