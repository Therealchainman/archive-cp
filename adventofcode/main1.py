from collections import *
from functools import *
from itertools import *
import sys
import re
import math
import string
import bisect
# sys.stdin = open("input.txt", "r")
sys.stdout = open("output.txt", "w")

def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        _, seeds = data[0].split(":")
        seeds = list(map(int, seeds.split()))
        ptr = 0
        maps = ["seed-to-soil", "soil-to-fertilizer", "fertilizer-to-water", "water-to-light", "light-to-temperature", "temperature-to-humidity", "humidity-to-location"]
        mappers = [[] for _ in range(len(maps))]
        for line in data[2:]:
            if line == "": 
                ptr += 1
                continue
            if maps[ptr] in line: continue
            dest, source, len_ = map(int, line.split())
            if maps[ptr] == "seed-to-soil":
                mappers[ptr].append((source, dest, len_))
            elif maps[ptr] == "soil-to-fertilizer":
                mappers[ptr].append((source, dest, len_))
            elif maps[ptr] == "fertilizer-to-water":
                mappers[ptr].append((source, dest, len_))
            elif maps[ptr] == "light-to-temperature":
                mappers[ptr].append((source, dest, len_))
            elif maps[ptr] == "temperature-to-humidity":
                mappers[ptr].append((source, dest, len_))
            else:
                mappers[ptr].append((source, dest, len_))
        for i in range(len(mappers)):
            mappers[i].sort()
        res = math.inf
        for i in range(0, len(seeds), 2):
            start = seeds[i]
            end = seeds[i] + seeds[i + 1]
            for seed in range(start, end):
                val = seed
                for i in range(len(mappers)):
                    j = bisect.bisect_right(mappers[i], (val, math.inf, math.inf)) - 1
                    if j == -1 or val > mappers[i][j][0] + mappers[i][j][2]: continue
                    else: val = mappers[i][j][1] + (val - mappers[i][j][0])
                res = min(res, val)
        print(res)      

if __name__ == '__main__':
    main()