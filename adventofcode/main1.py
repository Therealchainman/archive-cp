import re
from collections import defaultdict
import math
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        sensors = []
        beacons = []
        y = 2000000
        rowBeacons = set()
        manhattan = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        process = lambda s: int(re.findall(r'-?\d+', s)[0])
        for line in map(lambda x: x.split(), data):
            x1, y1 = map(lambda x: process(x), [line[2], line[3]])
            x2, y2 = map(lambda x: process(x), [line[8], line[9]])
            sensors.append((x1, y1))
            beacons.append((x2, y2))
            if y2 == y:
                rowBeacons.add(x2)
        ranges = defaultdict(int)
        for (x1, y1), (x2, y2) in zip(sensors, beacons):
            highest = manhattan(x1, y1, x2, y2) - manhattan(x1, y1, x1, y)
            if highest < 0: continue
            left = x1-highest
            right = x1+highest
            ranges[left] += 1
            ranges[right+1] -= 1
        rowBeacons = sorted(list(rowBeacons))
        delta = res = beacon_ptr = 0
        start = math.inf
        for key in sorted(ranges.keys()):
            if delta == 0:
                start = key
            delta += ranges[key]
            if delta == 0:
                res += key - start
                while beacon_ptr < len(rowBeacons) and rowBeacons[beacon_ptr] < key:
                    res -= (beacon_ptr >= start)
                    beacon_ptr += 1
        return res
if __name__ == "__main__":
    print(main())