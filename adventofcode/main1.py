import itertools
def main():
    with open('input.txt', 'r') as f:
        data = []
        lines = f.read().splitlines()
        for line in lines:
            data.append(list(line))
        R, C = len(data), len(data[0])
        intersection = '+'
        headings = {'>': (0, 1), '<': (0, -1), '^': (-1, 0), 'v': (1, 0)}
        paths = {0: 'left', 1: 'straight', 2: 'right'}
        mine_update = {'>': '-', '<': '-', '^': '|', 'v': '|'}
        carts = []
        coords = set()
        for r, c in itertools.product(range(R), range(C)):
            if data[r][c] in headings:
                carts.append((r, c, data[r][c], 0)) # (r, c, dir, turns)
                data[r][c] = mine_update[(data[r][c])]
                coords.add((r, c))
        while True:
            newCarts = [None]*len(carts)
            for i, (r, c, dir_, turns) in enumerate(carts):
                dr, dc = headings[dir_]
                nr = r + dr
                nc = c + dc
                if (nr, nc) in coords:
                    return ','.join(map(str, [nc, nr]))
                ndir = dir_
                nturns = turns
                if data[nr][nc] == intersection:
                    if paths[nturns] == 'left':
                        ndir = '^>v<'[">v<^".index(dir_)]
                    elif paths[nturns] == 'right':
                        ndir = '^>v<'["<^>v".index(dir_)]
                    nturns = (nturns + 1) % 3
                elif data[nr][nc] == '\\':
                    ndir = '<^>v'['^<v>'.index(dir_)]
                elif data[nr][nc] == '/':
                    ndir = '>v<^'['^<v>'.index(dir_)]
                newCarts[i] = (nr, nc, ndir, nturns)
                coords.discard((r, c))
                coords.add((nr, nc))
            carts = newCarts
            carts.sort()
        return '-1,-1'

if __name__ == "__main__":
    print(main())