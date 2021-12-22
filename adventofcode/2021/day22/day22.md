

# Part 2 

Part 1 is the same but it limits the input so you can get away with a more brute forcy solution.  

For part 2 you have such a large input you need to keep track of the cuboids appropriately.  

Basic idea is to store the cuboids in a counter, and keep count.  So when you turn a cuboid on, you turn on that cuboid in the counter
But you need to make sure that if there are any overlaps with previous cuboids that you remove those because when you add your current cuboid
it will be double counting the cells that overlap with other cuboids. 

Then if you are turning off you just want to remove the overlapping with other cuboids
```py
import re
from collections import namedtuple, Counter
class Instruction:
    def __init__(self, status, coords):
        self.status = status
        Range = namedtuple("Range", ['initial', 'final'])
        self.xrange = Range(coords[0], coords[1])
        self.yrange = Range(coords[2], coords[3])
        self.zrange = Range(coords[4], coords[5])
    def __repr__(self):
        return f"command: {self.status}, x = ({self.xrange.initial}, {self.xrange.final}), y = ({self.yrange.initial}, {self.yrange.final}, z = ({self.zrange.initial}, {self.zrange.final})"
class ReactorReboot:
    def __init__(self):
        self.instructions = None

    def data_loader(self):
        with open("inputs/input.txt","r") as f:
            return [Instruction(line.split()[0], list(map(int, re.findall(r'[-\d]+', line)))) for line in f.read().splitlines()]

    def run(self):
        self.instructions = self.data_loader()
        cuboids = Counter()
        for instruction in self.instructions:
            new_cuboids = Counter()
            nx0, nx1, ny0, ny1, nz0, nz1 = instruction.xrange.initial,instruction.xrange.final, instruction.yrange.initial, instruction.yrange.final, instruction.zrange.initial, instruction.zrange.final
            for (x0,x1,y0,y1,z0,z1), sgn in cuboids.items():
                bx0, bx1, by0, by1, bz0, bz1 = max(x0,nx0), min(x1,nx1), max(y0,ny0), min(y1,ny1), max(z0,nz0), min(z1,nz1)
                if bx0<=bx1 and by0<=by1 and bz0<=bz1:
                    new_cuboids[(bx0,bx1,by0,by1,bz0,bz1)] -= sgn
            if instruction.status=="on":
                new_cuboids[(nx0,nx1,ny0,ny1,nz0,nz1)] += 1
            cuboids.update(new_cuboids)
        return sum((x1-x0+1)*(y1-y0+1)*(z1-z0+1)*sgn for (x0,x1,y0,y1,z0,z1), sgn in cuboids.items())

if __name__ == '__main__':
    reactor = ReactorReboot()
    print(reactor.run())
```