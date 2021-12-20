
     
from collections import defaultdict
from itertools import permutations, product
 
 
class Point():
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
 
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y, self.z - other.z)
 
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
 
    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'
 
    def __repr__(self):
        return f'({self.x}, {self.y}, {self.z})'
 
    def __hash__(self):
        return hash(str(self))
 
    def manhattan_dist(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)
 
 
class Scanner():
    def __init__(self, detected_points=[], location=None):
        self.detected_points = detected_points
        self.location = location
 
 
def generate_rotations(points):
    rotations = []
 
    for rotation in permutations(['x', 'y', 'z']):
        for signs in product([-1, 1], repeat=3):
            single_rotation = []
 
            for point in points:
                axes = {'x': point.x, 'y': point.y, 'z': point.z}
                single_rotation.append(
                    Point(axes[rotation[0]] * signs[0], axes[rotation[1]] * signs[1], axes[rotation[2]] * signs[2]))
 
            rotations.append(single_rotation)
 
    return rotations
 
 
def check_same_shape(located_scanner, unlocated_scanner):
    for rotation in generate_rotations(unlocated_scanner.detected_points):
        counts = defaultdict(int)
 
        for point_1 in rotation:
            for point_2 in located_scanner.detected_points:
                counts[point_2 - point_1] += 1
 
        for k in counts:
            if counts[k] == 12:
                # print(k)
                return True, Point(k[0], k[1], k[2]), rotation
 
    return False, None, None
 
 
def convert_to_absolute(scanner_location, points):
    new = []
 
    for point in points:
        new.append(Point(point.x + scanner_location.x, point.y +
                   scanner_location.y, point.z + scanner_location.z))
    # print(new)
    return new
 
 
def locate_scanners(scanners):
    n = len(scanners)
    located_scanners = {
        0: Scanner(scanners[0].detected_points, Point(0, 0, 0))
    }
 
    while len(located_scanners) != n:
        for i in range(n):
            if i in located_scanners:
                continue
 
            unlocated_scanner = scanners[i]
 
            for j in located_scanners:
                located_scanner = located_scanners[j]
 
                valid, scanner_location, rotation = check_same_shape(
                    located_scanner, unlocated_scanner)
 
                if not valid:
                    continue
                # print("unlocated scanner:", i)
                # print("located scanner:", j)
                # print(scanner_location)
                # print(rotation)
                newly_located_scanner = Scanner(
                    convert_to_absolute(
                        scanner_location, rotation
                    ),
                    scanner_location
                )
 
                located_scanners[i] = newly_located_scanner
 
                break
 
    res = [None] * n
 
    for i in located_scanners:
        res[i] = located_scanners[i]
 
    return res
 
 
def solve_part_1(scanners):
    located_scanners = locate_scanners(scanners)
 
    points = set()
 
    for scanner in located_scanners:
        points.update(scanner.detected_points)
 
    return len(points)
 
 
def solve_part_2(scanners):
    located_scanners = locate_scanners(scanners)
 
    res = float('-inf')
    for i in range(len(located_scanners)):
        for j in range(i+1, len(located_scanners)):
            scanner_1 = located_scanners[i]
            scanner_2 = located_scanners[j]
 
            res = max(res, scanner_1.location.manhattan_dist(
                scanner_2.location))
 
    return res
 
 
if __name__ == '__main__':
    with open('inputs/input.txt') as f:
        scanners = []
 
        for scanner in f.read().split('\n\n'):
            detected_points = []
 
            for coords in scanner.split('\n')[1:]:
                x, y, z = [int(coord) for coord in coords.split(',')]
                detected_points.append(Point(x, y, z))
 
            scanners.append(Scanner(detected_points))
 
        # print(solve_part_1(scanners))
        print(solve_part_2(scanners))