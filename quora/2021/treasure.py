"""
Using loops with caching previous subproblems for the max treasures at cell i,j and the
count of paths at i,j with max treasures
"""
class Treasure:
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            self.n = int(f.readline())
            self.grid = [[int(x) for x in f.readline().strip()] for _ in range(self.n)]
        # self.n = int(input())
        # self.grid = [[int(x) for x in input().split()] for _ in range(self.n)]

    def run(self):
        self.data_loader()
        cache_max = [[0 for _ in range(self.n+1)] for _ in range(self.n+1)]
        cache_paths = [[0 for _ in range(self.n+1)] for _ in range(self.n+1)]
        # computing the max treasure at a cell
        for i in range(self.n):
            for j in range(self.n):
                cache_max[i+1][j+1] = max(cache_max[i][j+1], cache_max[i+1][j]) + self.grid[i][j]
        cache_paths[1][1]=1
        # computing the number of paths to the maximum treasure at any cell
        for i in range(self.n):
            for j in range(self.n):
                max_treasures = max(cache_max[i][j+1], cache_max[i+1][j])
                if cache_max[i][j+1]==max_treasures:
                    cache_paths[i+1][j+1] += cache_paths[i][j+1]
                if cache_max[i+1][j]==max_treasures:
                    cache_paths[i+1][j+1] += cache_paths[i+1][j]
        print(cache_max[self.n][self.n], cache_paths[self.n][self.n])
if __name__ == '__main__':
    Treasure().run()