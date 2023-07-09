class Solution:
    def countBlackBlocks(self, R: int, C: int, coordinates: List[List[int]]) -> List[int]:
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        neighborhood = lambda r, c: [(r - 1, c), (r - 1, c - 1), (r, c - 1), (r, c)]
        black_counter = Counter()
        for r, c in coordinates:
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc): continue
                cell = nr + nr * nc
                black_counter[cell] += 1
        counts = [0] * 5
        for cnt in black_counter.values():
            counts[cnt] += 1
        counts[0] = (R - 1) * (C - 1) - sum(counts)
        return counts