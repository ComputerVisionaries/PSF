from load_puzzle import load_puzzle, generate_edges, extractPieceWithCoords
from fitness.RelativePosition import RelativePosition
from fitness.similarity_puzzle import similarity

import matplotlib.pyplot as plt

if __name__ == '__main__':
    pieces = load_puzzle()

    for piece in pieces:
        extractPieceWithCoords(piece.image)
        # points1 = piece.getSide('r', 1000)
        # plt.imshow(piece.image)
        # rows = [pt[0] for pt in points1]
        # cols = [pt[1] for pt in points1]
        # plt.plot(cols, rows, 'ro', ms=0.2)
        # plt.show()

    # similarity_value = similarity(piece1, piece2, RelativePosition.LEFT_RIGHT)
    # print(similarity_value)
