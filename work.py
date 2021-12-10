"""
Brute force algorithm that uses dfs with backtracking to tile a board with polyominoes (dominoes and trominoes).
"""
import numpy as np
import sys
sys.stdout = open('output.txt', 'w')
class Polyomino:
    def __init__(self, n):
        """
        The board is a 2 x n matrix 
        """
        self.board = np.zeros((2,n), dtype=int)
        self.count = 1 # counter for filling the board with distinct dominos
        self.n = n # size of the board
        self.countFilled = 0 # counter for the number of filled tiles
        self.numTilings = 0 # the number of ways to tile the board
    def inBounds(self, x, y):
        """
        Checks if the location is out of bounds
        """
        return 0 <= x < 2 and 0 <= y < self.n
    def empty(self, x, y):
        """
        Checks if the location is empty
        """
        return self.board[x][y] == 0
    def canTile(self, tiles):
        """
        Checks if the tile is valid
        """
        return all(self.inBounds(x,y) and self.empty(x,y) for x,y in tiles)
    def place(self, tiles):
        """
        Places the tiles on the board
        """
        for x,y in tiles:
            self.board[x][y] = self.count
    def remove(self, tiles):
        """
        Removes the tiles from the board
        """ 
        for x,y in tiles:
            self.board[x][y] = 0
    def main(self, i, j):
        """
        Main function
        """
        if self.countFilled == self.n*2:
            self.numTilings += 1
            print("======board======")
            print(self.board)
            return
        if not self.inBounds(i,j):
            return
        for tiles in [[(i,j),(i+1,j)],[(i,j),(i,j+1)], [(i,j),(i+1,j),(i,j+1)],[(i,j),(i-1,j),(i,j+1)], [(i,j),(i-1,j),(i,j-1)],[(i,j),(i,j-1),(i+1,j)]]:
            if self.canTile(tiles):
                self.place(tiles) # a function to place the tiles
                self.countFilled += len(tiles)
                self.count += 1
                if i == 0:
                    self.main(i+1,j)
                else:
                    self.main(0,j+1)
                self.remove(tiles) # a function to remove the tiles
                self.countFilled -= len(tiles)
                self.count -= 1
        if i == 0:
            self.main(i+1,j)
        else:
            self.main(0,j+1)
        

    def run(self):
        """
        Starts generating the configurations from the upper left corner
        """
        self.main(0,0)
if __name__ == '__main__':
    for i in range(1,6):
        p = Polyomino(i)
        p.run()
        print(f"The number of configrations = {p.numTilings} for a board of size {2}x{i}")
sys.stdout.close()