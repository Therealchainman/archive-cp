from collections import defaultdict, deque
from parse import compile
import math

pat = compile("{:d},{:d},{:d}")

def main():
    with open("input.txt", 'r') as f:
        data = f.read().splitlines()
        cubes = set()
        mx, mn = -math.inf, math.inf
        for x, y, z in map(lambda x: pat.parse(x), data):
            cubes.add((x, y, z))
            mx = max([mx, x, y, z])
            mn = min([mn, x, y, z])
        res = 0
        neighborhood = lambda x, y, z: [(x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)]
        def in_interior(x, y, z):
            queue = deque([(x, y, z)])
            vis = set([(x,y,z)])
            while queue:
                x, y, z = queue.popleft()
                for nx, ny, nz in neighborhood(x, y, z):
                    if (nx, ny, nz) in cubes or (nx, ny, nz) in vis: continue
                    if nx > mx or ny > mx or nz > mx or nx < mn or ny < mn or nz < mn:
                        return False
                    queue.append((nx, ny, nz))
                    vis.add((nx,ny,nz))
            return True
        for x, y, z in cubes:
            for nx, ny, nz in neighborhood(x, y, z):
                if (nx, ny, nz) in cubes or in_interior(nx, ny, nz): continue
                res += 1
        return res

if __name__ == '__main__':
    print(main())
