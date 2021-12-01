"""
Part 2
"""
if __name__ == "__main__":
    with open("inputs/input1.txt", "r") as f:
        data = list(map(int, f.read().splitlines()))
        print(sum(1 for num1, num4 in zip(data, data[3:]) if num4>num1))
