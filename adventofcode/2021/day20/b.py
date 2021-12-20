INF = 100000
class imageEnhancement:
    def __init__(self):
        self.algorithm = None
        self.data = None

    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            lst = f.read().split('\n')
            self.algorithm = lst[0]
            data = lst[2:]
            self.data = set()
            for i in range(len(data)):
                for j in range(len(data[0])):
                    if data[i][j] == '#':
                        self.data.add((i, j))

    def enhance(self, iteration):
        row_min, row_max, col_min, col_max = INF, -INF, INF, -INF
        rows, cols = zip(*self.data)
        row_min, row_max, col_min, col_max = min(rows), max(rows), min(cols), max(cols)
        new_lights = set()
        def improve(row, col):
            binary_value = ""
            for i in range(row - 1, row + 2):
                for j in range(col - 1, col + 2):
                    if i<row_min or i>row_max or j<col_min or j>col_max:
                        binary_value += str(iteration)
                        continue
                    binary_value += '1' if (i,j) in self.data else '0'
            i = int(binary_value, 2)
            if self.algorithm[i] == '#':
                new_lights.add((row, col))
        for i in range(row_min-2, row_max+3):
            for j in range(col_min-2,col_max+3):
                improve(i, j)
        self.data = new_lights
    def run(self):
        self.data_loader()
        for i in range(50):
            self.enhance(i%2)
        print(len(self.data))

if __name__ == "__main__":
    imageEnhancement().run()
