from collections import defaultdict
from parse import compile

pat = compile("{:d},{:d},{:d}")

def main():
    with open("input.txt", 'r') as f:
        data = f.read().splitlines()
        cubes = set()
        for x, y, z in map(lambda x: pat.parse(x), data):
            cubes.add((x, y, z))
        res = 0
        neighborhood = lambda x, y, z: [(x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)]
        for x, y, z in cubes:
            for nx, ny, nz in neighborhood(x, y, z):
                if (nx, ny, nz) in cubes: continue
                res += 1
        return res

if __name__ == '__main__':
    print(main())
