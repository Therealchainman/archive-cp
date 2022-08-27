import sys
from collections import deque
from itertools import product

problem = sys.argv[0].split('.')[0]
validation = 'validation_'

class Solution:
    def __init__(self):
        result = []
        self.fileStream = open(f'inputs/{problem}_{validation}input.txt', 'r')
        T = int(self.fileStream.readline())
        for t in range(1,T+1):
            result.append(f'Case #{t}: {self.main()}')
        with open(f'outputs/{problem}_output.txt', 'w') as f:
            f.write('\n'.join(result))

    def grow(self):
        for r in range(self.R):
            row = [self.rock]*self.C
            for c in range(self.C):
                if self.arr[r][c] == self.blocked:
                    row[c] = self.empty
                if self.arr[r][c] == self.empty or self.arr[r][c] == self.tree:
                    row[c] = self.tree
            yield ''.join(row)

    def in_bounds(self, r, c):
        return 0<=r<self.R and 0<=c<self.C

    def eligible_cells(self, r, c):
        return self.arr[r][c] == self.tree or self.arr[r][c] == self.empty

    def neighbors(self, r, c):
        return list(filter(lambda x: self.in_bounds(x[0], x[1]) and self.eligible_cells(x[0], x[1]), [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]))

    def get_rocks(self):
        for r, c in product(range(self.R), range(self.C)):
            if self.arr[r][c] == self.rock:
                yield (r, c)

    def find_blocks(self):
        queue = deque(list(self.get_rocks()))
        while queue:
            r, c = queue.popleft()
            for nr, nc in self.neighbors(r,c):
                if len(self.neighbors(nr,nc)) < 2:
                    if self.arr[nr][nc] == self.tree: return False
                    self.arr[nr][nc] = self.blocked
                    queue.append((nr, nc))
        return True

    def main(self):
        self.R, self.C = map(int,self.fileStream.readline().split())
        self.arr = [list(self.fileStream.readline().rstrip()) for _ in range(self.R)]
        if all(t!=self.tree for row in self.arr for t in row):
            return 'Possible\n' + '\n'.join(''.join(row) for row in self.arr)
        if self.R == 1 or self.C == 1 or not self.find_blocks():
            return 'Impossible'
        return 'Possible\n' + '\n'.join(self.grow())

if __name__ == '__main__':
    Solution()