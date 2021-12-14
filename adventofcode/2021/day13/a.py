class TransparentOrigami:
    def __init__(self):
        self.data = set()
        self.maxY = 0
        self.maxX = 0
        self.folds = []
    def dataLoader(self):
        with open("inputs/input.txt", "r") as f:
            points, folds = f.read().split('\n\n')
            self.data = {tuple(map(int,points.split(','))) for points in points.split('\n')}
            self.folds = [(fold[11], int(fold[13:])) for fold in folds.split('\n')]
    def fold(self, axis, n):
        if axis == 'x':
            return {(2*n-x, y) if x > n else (x,y) for x,y in self.data}
        return {(x, 2*n-y) if y > n else (x,y) for x,y in self.data}
    def run(self):
        self.dataLoader()
        for axis, n in self.folds:
            self.data = self.fold(axis,n)
            return len(self.data) # just for part 1 cause it needs to just do one fold
        return len(self.data)

if __name__ == '__main__':
    s = TransparentOrigami()
    print(s.run())
    