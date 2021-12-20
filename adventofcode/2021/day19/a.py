import numpy as np
from collections import defaultdict, deque
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

    def matches(self, scan1, scan2, rot, initial_beacon1, initial_beacon2, dx, dy, dz):
        # create a set of all the beacons in scan1
        seen = set(map(tuple, self.data[scan1]))
        seen.remove(tuple(self.data[scan1][initial_beacon1]))
        numMatches = 1
        for i, beacon in enumerate(self.data[scan2]):
            if i == initial_beacon2:
                continue
            beacon = np.matmul(rot, beacon)
            cand = (beacon[0]-dx, beacon[1]-dy, beacon[2]-dz)
            if cand in seen:
                numMatches += 1
                seen.remove(cand)
        return numMatches
    def findsMatch(self, i, j):
        for rot_matrix in self.rotations():
            for id1, beacon1 in enumerate(self.data[i]):
                for id2, beacon2 in enumerate(self.data[j]):
                    beacon2 = np.matmul(rot_matrix, beacon2)
                    # assume beacon1 and beacon2 are the same beacons with the current rotation of scanner j about scanner i
                    dx, dy, dz = beacon2[0]-beacon1[0], beacon2[1]-beacon1[1], beacon2[2]-beacon1[2]
                    # I would have to subtbeacon2[0ract dx, dy and dz from beacon2 for it to have the same coordinates as beacon1
                    if self.matches(i, j, rot_matrix, id1, id2, dx, dy, dz)>=12:
                        self.delta = (dx, dy, dz)
                        return True
        return False
    def run(self):
        self.data_loader()
        numScanners = len(self.data)
        graph = defaultdict(list)
        for i in range(numScanners):
            for j in range(i+1,numScanners):
                if self.findsMatch(i,j):
                    graph[i].append((j, *tuple(map(lambda x: -x, self.delta))))
                    graph[j].append((i, *self.delta))
        # dfs through the graph to check that it gets all of them
        q = deque([(0,0,0,0)])
        vis = [False for _ in range(numScanners)]
        vis[0]=True
        dist = [tuple() for _ in range(numScanners)]
        dist[0] = (0,0,0)
        print(graph)
        while q:
            scanner,dx,dy,dz = q.popleft()
            for nei, ddx, ddy, ddz in graph[scanner]:
                if not vis[nei]:
                    dist[nei] = (dx+ddx, dy+ddy, dz+ddz)
                    q.append((nei,dx+ddx,dy+ddy,dz+ddz))
                    vis[nei]=True
        print(dist)
                    




if __name__ == '__main__':
    bs = BeaconScanner()
    bs.run()