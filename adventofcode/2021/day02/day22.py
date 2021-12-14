if __name__ == '__main__':
    with open("inputs/input1.txt", "r") as f:
        arr = list(map(lambda x: (x.split()[0], int(x.split()[1])), f.read().splitlines()))
        hor, depth, aim = 0, 0, 0
        for dir, magnitude in arr:
            hor += magnitude if dir == 'forward' else 0
            depth += (magnitude*aim) if dir == 'forward' else 0
            aim += magnitude *(1 if dir == 'down' else -1 if dir=='up' else 0)
        print(hor*depth)
