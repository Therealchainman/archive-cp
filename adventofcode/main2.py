from itertools import *
class Fabric:
    def __init__(self, fabric):
        self.id = int(fabric.split()[0][1:])
        self.x = int(fabric.split()[2].split(',')[0])
        self.y = int(fabric.split()[2].split(',')[1][:-1])
        self.width = int(fabric.split()[3].split('x')[0])
        self.height = int(fabric.split()[3].split('x')[1])
        self.x2 = self.x + self.width
        self.y2 = self.y + self.height
    def __repr__(self):
        return f'id: {self.id} x: {self.x} y: {self.y} width: {self.width} height: {self.height}'

def main():
    with open('input.txt', 'r') as f:
        data = list(map(Fabric, f.read().splitlines()))
        for f1 in data:
            if all(max(0, min(f1.x2, f2.x2) - max(f1.x, f2.x))*max(0, min(f1.y2, f2.y2) - max(f1.y, f2.y)) == 0 for f2 in data if f1 != f2):
                return f1.id
        return -1
if __name__ == "__main__":
    print(main())