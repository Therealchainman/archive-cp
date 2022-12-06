from collections import defaultdict, deque, Counter
from math import inf
import re
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        pattern = r"\d+"
        queue = deque()
        minDist = defaultdict(lambda: inf)
        area = Counter()
        lastid = {}
        threshold = 100 # trial and error
        for id, (x, y) in enumerate(map(lambda coords: map(int, re.findall(pattern, coords)), data)):
            queue.append((id, x, y, 0))
            minDist[(x, y)] = 0
            area[id] += 1
        neighborhood = lambda x, y: ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
        while queue:
            id, x, y, dist = queue.popleft()
            if dist == threshold:
                area[id] = -inf
                continue
            for nx, ny in neighborhood(x, y):
                state = (nx, ny)
                ndist = dist + 1
                if ndist >= minDist[state]: 
                    if ndist == minDist[state]:
                        if lastid.get(state, id) != id:
                            area[lastid[state]] -= 1
                            lastid.pop(state)
                    continue
                queue.append((id, nx, ny, ndist))
                area[id] += 1
                lastid[state] = id
                minDist[state] = ndist
        return max(area.values())
if __name__ == "__main__":
    print(main())