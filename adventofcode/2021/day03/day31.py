if __name__ == '__main__':
    with open("inputs/input.txt", "r") as f:
        gammaBits = list(map(lambda x: '1' if x.count('1')>x.count('0') else '0', zip(*f.read().splitlines())))
        gamma = int("".join(gammaBits), 2)
        n = len(gammaBits)
        epsilon = gamma ^ ((1<<n)-1)
        print(gamma*epsilon)
