import numpy as np
from collections import defaultdict, deque, Counter
class BeaconScanner:
    def __init__(self):
        self.data = None
        self.delta = None # The relative position between two scanners

    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            self.data = [[np.fromiter(map(int, coords.split(',')), dtype=int) for coords in lst.split('\n')[1:]] for lst in f.read().split('\n\n')]
    """
    TODO: Write this in a numpy method that generates the rotation matrices without hard coding
    Quick solution is to hard code the rotation matrices
    """
    def rotations(self):
        yield np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=int)
        yield np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=int)
        yield np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=int)
        yield np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=int)
        yield np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=int)
        yield np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=int)
        yield np.array([[0,1,0],[1,0,0],[0,0,-1]], dtype=int)
        yield np.array([[0,0,-1],[1,0,0],[0,-1,0]], dtype=int)
        yield np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=int)
        yield np.array([[-1,0,0],[0,0,-1],[0,-1,0]], dtype=int)
        yield np.array([[-1,0,0],[0,1,0],[0,0,-1]], dtype=int)
        yield np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=int)
        yield np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=int)
        yield np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=int)
        yield np.array([[0,-1,0],[-1,0,0],[0,0,-1]], dtype=int)
        yield np.array([[0,0,-1],[-1,0,0],[0,1,0]], dtype=int)
        yield np.array([[0,0,-1],[0,1,0],[1,0,0]], dtype=int)
        yield np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=int)
        yield np.array([[0,0,1],[0,-1,0],[1,0,0]], dtype=int)
        yield np.array([[0,-1,0],[0,0,-1],[1,0,0]], dtype=int)
        yield np.array([[0,0,-1],[0,-1,0],[-1,0,0]], dtype=int)
        yield np.array([[0,-1,0],[0,0,1],[-1,0,0]], dtype=int)
        yield np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=int)
        yield np.array([[0,1,0],[0,0,-1],[-1,0,0]], dtype=int)

    def rotation_beacons(self, scanner):
        for rotation in self.rotations():
            beacons = []
            for beacon in self.data[scanner]:
                beacons.append(np.matmul(rotation, beacon))
            yield beacons
            
    def matches(self, scan1, scan2, rot, initial_beacon1, initial_beacon2, dx, dy, dz):
        # create a set of all the beacons in scan1
        seen = set(map(tuple, self.data[scan1]))
        numMatches = 1
        for i, beacon in enumerate(self.data[scan2]):
            if i == initial_beacon2:
                continue
            beacon = np.matmul(rot, beacon)
            cand = (beacon[0]-dx, beacon[1]-dy, beacon[2]-dz)
            if cand in seen:
                numMatches += 1
        return numMatches

    def findsMatch(self, located_scanner, unlocated_scanner):
        for beacons in self.rotation_beacons(unlocated_scanner):
            count = Counter()
            for beacon2 in self.data[located_scanner]:
                for beacon in beacons:
                    new_beacon = beacon-beacon2
                    count[tuple(new_beacon)] += 1
                    for k, cnt in count.items():
                        if cnt==12:
                            self.data[unlocated_scanner] = set()
                            for beacon in beacons:
                                self.data[unlocated_scanner].add(tuple(beacon-beacon2))
                            return True
        return False

    def run(self):
        self.data_loader()
        numScanners = len(self.data)
        located_scanners = set([0])
        unlocated_scanners = set(range(1,numScanners))
        while len(unlocated_scanners)>0:
            print(len(unlocated_scanners))
            for i in range(numScanners):
                if i in located_scanners:
                    continue
                for j in located_scanners:
                    if self.findsMatch(j,i):
                        located_scanners.add(i)
                        unlocated_scanners.remove(i)
                        break
        # add up all the beacons that are measured relative to scanner 0 from all the other scanners
        # This will be the total count of beacons in the water
        res = set()
        for i in range(numScanners):
            for beacon in self.data[i]:
                res.add(tuple(beacon))
        print(len(res))
                    




if __name__ == '__main__':
    bs = BeaconScanner()
    bs.run()
