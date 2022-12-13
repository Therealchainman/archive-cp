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
        coords = {}
        cnt = 0
        for r, c in itertools.product(range(R), range(C)):
            if data[r][c] in headings:
                carts.append((r, c, cnt, data[r][c], 0)) # (r, c, dir, turns)
                coords[(r, c)] = cnt
                cnt += 1
                data[r][c] = mine_update[(data[r][c])]
        crashed_carts = set()
        while cnt > 1:
            newCarts = []
            for i, (r, c, id, dir_, turns) in enumerate(carts):
                if id in crashed_carts: continue
                dr, dc = headings[dir_]
                nr = r + dr
                nc = c + dc
                if (nr, nc) in coords:
                    crashed_carts.add(coords[(nr, nc)])
                    del coords[(nr, nc)]
                    del coords[(r, c)]
                    cnt -= 2
                    continue
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
                newCarts.append((nr, nc, id, ndir, nturns))
                del coords[(r, c)]
                coords[(nr, nc)] = id
            carts = newCarts
            carts.sort()
        for i, (r, c, id, dir_, turns) in enumerate(carts):
            if id not in crashed_carts:
                return ','.join(map(str, [c, r]))

if __name__ == "__main__":
    print(main())