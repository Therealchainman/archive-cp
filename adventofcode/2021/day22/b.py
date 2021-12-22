"""
on x=-20..26,y=-36..17,z=-47..7
"""
import re
from collections import namedtuple
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

    def feasible(self, instruct):
        def in_boundary(rng):
            return rng.initial>=-50 and rng.final<=50
        return in_boundary(instruct.xrange) and in_boundary(instruct.yrange) and in_boundary(instruct.zrange)

    def run(self):
        self.instructions = self.data_loader()
        cuboids = set()
        for instruction in self.instructions:
            if self.feasible(instruction):
                for x in range(instruction.xrange.initial, instruction.xrange.final+1):
                    for y in range(instruction.yrange.initial, instruction.yrange.final+1):
                        for z in range(instruction.zrange.initial, instruction.zrange.final+1):
                            if instruction.status=="on":
                                cuboids.add((x,y,z))
                            else:
                                cuboids.discard((x,y,z))
        return len(cuboids)

if __name__ == '__main__':
    reactor = ReactorReboot()
    print(reactor.run())