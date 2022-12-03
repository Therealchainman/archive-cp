class Fabric:
    def __init__(self, fabric):
        self.id = fabric.split()[0]
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
        data = f.read().splitlines()
        visited = [[0]*1000 for _ in range(1000)]
        res = 0
        for fabric in map(Fabric, data):
            for x, y in product(range(fabric.x, fabric.x2), range(fabric.y, fabric.y2)):
                res += (visited[x][y]==1)
                visited[x][y] += 1
        return res
if __name__ == "__main__":
    print(main())