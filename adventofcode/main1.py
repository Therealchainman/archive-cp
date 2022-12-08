from math import inf
def main():
    with open('input.txt', 'r') as f:
        data = []
        lines = f.read().splitlines()
        for line in lines:
            data.append([int(x) for x in line])
        n = len(data)
        visible = set()
        for r in range(n):
            maxVal = -inf
            for c in range(n):
                if data[r][c] > maxVal:
                    visible.add((r, c))
                    maxVal = data[r][c]
            maxVal = -inf
            for c in reversed(range(n)):
                if data[r][c] > maxVal:
                    visible.add((r, c))
                    maxVal = data[r][c]
        for c in range(n):
            maxVal = -inf
            for r in range(n):
                if data[r][c] > maxVal:
                    visible.add((r, c))
                    maxVal = data[r][c]
            maxVal = -inf
            for r in reversed(range(n)):
                if data[r][c] > maxVal:
                    visible.add((r, c))
                    maxVal = data[r][c]
        return len(visible)
if __name__ == "__main__":
    print(main())