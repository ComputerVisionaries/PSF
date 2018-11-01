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

def roll_x(puzzle, dimensions, shift):
    width, height = dimensions
    curr = puzzle.copy()
    for _ in range(shift):
        new = np.zeros(curr.shape)
        for i in range(1, height+1):
            new[(i-1)*width] = curr[i*width-1]
            new[(i-1)*width+1:i*width] = curr[(i-1)*width:i*width-1]
        curr = new
    return curr

def roll_y(puzzle, dimensions, shift):
    curr = puzzle.copy()
    for i in range(shift):
        width, height = dimensions
        curr = np.concatenate([curr[width:], curr[:width]])
    return curr

def randomNeighbor(pieces, dimensions):
    width, height = dimensions
    if np.random.random() < 0.02:
        if np.random.random() < 0.5:
            return roll_x(pieces, dimensions, np.random.randint(1, width))
        else:
            return roll_y(pieces, dimensions, np.random.randint(1, width))
    i = np.random.randint(len(pieces))
    j = np.random.randint(len(pieces) - 1)
    # ensures distinct, unbiased samples of i and j
    if j >= i:
        j += 1
    copy = pieces.copy()
    copy[i], copy[j] = copy[j].copy(), copy[i].copy()
    return copy

def simulated_annealing_solver(pieces, dimensions, similarityFunction, kmax, t0):
    currConfig = np.random.permutation(pieces)
    currCost = cost(currConfig, dimensions, similarityFunction)
    bestConfig, bestCost = currConfig.copy(), currCost
    T = t0
    for k in range(kmax):
        T = t0 / np.log(k + 0.01)
        # alternatives:
        # T = t0 / (k + 0.01)
        # T = t0 * 0.99 ** k
        old = currCost
        cfg = randomNeighbor(currConfig, dimensions)
        cst = cost(cfg, dimensions, similarityFunction)
        if cst < bestCost:
            bestCost, bestConfig = cst, cfg
            print("---> new best cst=" + str(cst))
        p = 1.0
        if cst > currCost:
            p = np.exp(-(cst - currCost) / T)
        if p > np.random.random():
            currCost, currConfig = cst, cfg
            print("Swapped image @ k=" + str(k))
    print("Simulated annealing finished with cost=%.2f" % currCost)
    return (bestCost, bestConfig)

if __name__ == "__main__":
    full_img = plt.imread("../images/baboon.jpg")
    n = 3
    puzzle, _ = flatten_image(full_img, int(512 / n))
    optimalCost = cost(puzzle, (n, n), similarity.rgb_similarity)
    ax = plt.subplot(1, 1, 1)
    ax.set_title("cost=%.2f" % optimalCost)
    plt.imshow(inflate_image(puzzle, n, n))
    plt.show()
    sq_iters = 2
    results = sorted([simulatedAnnealing(puzzle, (n, n), similarity.rgb_similarity, 5000, 10.0) for _ in range(sq_iters ** 2)], key=lambda x: x[0])
    for i in range(sq_iters ** 2):
        ax = plt.subplot(sq_iters, sq_iters, i + 1)
        ax.set_title("cost=%.2f" % results[i][0])
        plt.imshow(inflate_image(results[i][1], n, n))
    plt.tight_layout()
    plt.show()
