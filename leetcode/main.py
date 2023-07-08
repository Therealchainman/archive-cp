class Solution:
    def countBlackBlocks(self, m: int, n: int, coordinates: List[List[int]]) -> List[int]:
        R, C = m, n
        n = len(coordinates)
        coordinates = set(coordinates)
        neighborhood = lambda r, c: [(r - 1, c), (r - 1, c - 1), (r, c - 1), (r, c)]
        neighborhood2 = lambda r, c: [(r + 1, c), (r + 1, c + 1), (r, c + 1), (r, c)]
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        total = (R - 1) * (C - 1)
        vis = set()
        counts = [0] * 5
        black_rocks = lambda r, c: sum(1 for r, c in neighborhood2(r, c) if in_bounds(r, c) and (r, c) in coordinates)
        for r, c in coordinates:
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) and (nr, nc) in vis: continue
                vis.add((nr, nc))
                cnt = black_rocks(nr, nc)
                counts[cnt] += 1
        counts[0] = total - sum(counts)
        return counts