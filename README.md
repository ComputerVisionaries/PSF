# PSF
Puzzle Solver Framework

#Dependencies:
- Python 2
- numpy
- scipy
    


#Structure:
- fitness: holds different fitness functions
    + Edge Difference: L2 norm of difference between the edges of 2 puzzle pieces
- images: jpg images to be used as puzzles
- puzzle: holds representation of puzzle
- solver: optimization algorithm frameworks
    + Genetic Algorithm
    + Randomized Hill Climbing
    + Simulated Annealing
- util: common tools to be used

image_scrambler.py : turns a target image into a puzzle (outputs puzzle.jpg into root dir)
run.py : currently only runs GA solver on puzzle.jpg (outputs solved_puzzle.jpg in root)



#Example Usage
python image_scrambler.py<br />
python run.py<br />
"Output can be seen in puzzle.py and solved_puzzle.jpg"