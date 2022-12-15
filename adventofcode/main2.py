from collections import defaultdict
import re
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        sensors = []
        beacons = []
        manhattan = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        process = lambda s: int(re.findall(r'-?\d+', s)[0])
        for line in map(lambda x: x.split(), data):
            x1, y1 = map(lambda x: process(x), [line[2], line[3]])
            x2, y2 = map(lambda x: process(x), [line[8], line[9]])
            sensors.append((x1, y1))
            beacons.append((x2, y2))
        limit = 4000000
        for y in range(0, limit):
            ranges = defaultdict(int)
            ranges[0] = 0
            for (x1, y1), (x2, y2) in zip(sensors, beacons):
                highest = manhattan(x1, y1, x2, y2) - manhattan(x1, y1, x1, y)
                if highest < 0: continue
                left = max(0, x1-highest)
                right = min(limit, x1+highest)
                ranges[left] += 1
                ranges[right+1] -= 1
            cur = 0
            for key in sorted(ranges.keys()):
                cur += ranges[key]
                if cur == 0 and key != limit+1:
                    return key*4000000+y
        return -1
if __name__ == "__main__":
    print(main())