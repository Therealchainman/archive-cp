"""
part 1
"""
if __name__ == "__main__":
    with open("inputs/input1.txt", "r") as f:
        data = list(map(int,f.read().splitlines()))
        print(sum(1 for prev, num in zip(data, data[1:]) if num>prev))