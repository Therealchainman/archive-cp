from collections import defaultdict
# sys.stdout = open('outputs/output.txt', 'w')

class Solver:
    def __init__(self):
        self.data = set()
        self.maxY = 0
        self.maxX = 0
        self.instructions = []
    def dataLoader(self):
        with open("inputs/input.txt", "r") as f:
            while True:
                line = f.readline().strip().split(',')
                if len(line)<2:
                    break
                self.data.add((int(line[0]), int(line[1])))
                self.maxX = max(self.maxX, int(line[0]))
                self.maxY = max(self.maxY, int(line[1]))
            while True:
                line = f.readline().strip()
                if not line:
                    break
                instruction = self.parse(line)
                foldLine = int(line[line.find('=')+1:])
                self.instructions.append((instruction, foldLine))
    def parse(self, line):
        return 1 if line.find('y')>=0 else 0
    def foldUp(self,y):
        for yi in range(y+1,self.maxY+1):
            abovePivot = yi-y
            for xi in range(self.maxX+1):
                if (xi,yi) in self.data:
                    self.data.remove((xi,yi))
                    nr = y-abovePivot
                    self.data.add((xi,nr))
        self.maxY = y
    def foldRight(self, x):
        for xi in range(x+1,self.maxX+1):
            rightPivot = xi-x
            # print(rightPivot)
            for yi in range(self.maxY+1):
                # print(rightPivot)
                if (xi,yi) in self.data:
                    # print(xi,yi)
                    self.data.remove((xi,yi))
                    nc = x-rightPivot
                    self.data.add((nc,yi))
                    # print(xi,nc)
        self.maxX = x
    
    def run(self):
        self.dataLoader()
        for instruction, foldLine in self.instructions:
            if instruction == 1:
                self.foldUp(foldLine)
            else:
                self.foldRight(foldLine)
            print(len(self.data))
        return len(self.data)

if __name__ == '__main__':
    s = Solver()
    s.run()
    