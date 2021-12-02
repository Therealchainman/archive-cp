from collections import namedtuple
if __name__ == '__main__':
    with open("inputs/input1.txt", "r") as f:
        arr = list(map(lambda x: (x.split()[0], int(x.split()[1])), f.read().splitlines()))
        hor = sum(magnitude for direction, magnitude in arr if direction in ['forward'])
        depth = sum(magnitude*(1 if direction=='down' else -1) for direction, magnitude in arr if direction in ['up', 'down'])
        print(hor*depth)
