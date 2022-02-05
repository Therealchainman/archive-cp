"""
The brute force solution seems reasonable by just updating the value for every node

But I can't seem to come up with a more optimized method, so I'll try the brute force way
and then see if I get any ideas. 
"""
import math

class DataCenter:
    def __init__(self):
        self.n = 0
        self.points = []

    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            self.n = int(f.readline())
            for line in f:
                x, y = map(int, line.split())
                self.points.append((x, y))
        # self.n = int(input())
        # for _ in range(self.n):
        #     x, y = map(int, input().split())
        #     self.points.append((x, y))

    def run(self):
        self.data_loader()
        lowest = math.inf
        idx = 0
        for i, (x, y) in enumerate(self.points,1):
            # center point
            cost = 0
            for nx, ny in self.points:
                if x==nx and y==ny: continue
                cost += max(abs(x-nx), abs(y-ny))
            if cost < lowest:
                lowest = cost
                idx = i
        print(idx)


if __name__ == '__main__':
    dc = DataCenter()
    dc.run()