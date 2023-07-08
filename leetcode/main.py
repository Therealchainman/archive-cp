class Solution:
    def countBlackBlocks(self, R: int, C: int, coordinates: List[List[int]]) -> List[int]:
        coordinates = set([(r, c) for r, c in coordinates])
        neighborhood = lambda r, c: [(r - 1, c), (r - 1, c - 1), (r, c - 1), (r, c)]
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        base_in_bounds = lambda r, c: 0 <= r < R - 1 and 0 <= c < C - 1
        base_neighborhood = lambda r, c: [(r + 1, c), (r + 1, c + 1), (r, c + 1), (r, c)] 
        submatrix_search = lambda r, c: sum(1 for r, c in base_neighborhood(r, c) if in_bounds(r, c) and (r, c) in coordinates)
        vis = set()
        counts = [0] * 5
        for r, c in coordinates:
            for nr, nc in neighborhood(r, c):
                if not base_in_bounds(nr, nc) or (nr, nc) in vis: continue
                vis.add((nr, nc))
                cnt = submatrix_search(nr, nc)
                counts[cnt] += 1
        counts[0] = (R - 1) * (C - 1) - sum(counts)
        return counts