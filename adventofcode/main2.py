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
THRESHOLD = 10_000_000_000
# THRESHOLD = 110
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
            new_map = []
            ptr = 0
            for (source, dest, len_) in mappers[i]:
                start = source
                end = source + len_ - 1
                if ptr < start:
                    new_map.append((ptr, ptr, start - ptr))
                new_map.append((source, dest, len_))
                ptr = end + 1
            start = new_map[-1][0]
            end = new_map[-1][0] + new_map[-1][2] - 1
            if end < THRESHOLD:
                new_map.append((end + 1, end + 1, THRESHOLD - end))
            mappers[i] = new_map
        for ma in mappers:
            print(ma)
            prev = -1
            for s, d, l in ma:
                assert s == prev + 1, f"s = {s} prev = {prev}"
                prev = s + l - 1
        res = math.inf 
        for i in range(0, len(seeds), 2):
            start = seeds[i]
            end = seeds[i] + seeds[i + 1] - 1
            ranges = [(start, end)]
            i = 0
            for ma in mappers:
                i += 1
                new_ranges = []
                for s, e in ranges:
                    for j in range(len(ma)):
                        st = ma[j][0]
                        en = ma[j][0] + ma[j][2] - 1
                        if s == st:
                            print(f"s = {s:,}, e = {e:,}, st = {st:,}, en = {en:,}, destination = {ma[j][1]:,}")
                        if st <= s <= e <= en:
                            if i == len(mappers):
                                print(f"s = {s:,}, e = {e:,}, st = {st:,}, en = {en:,}, destination = {ma[j][1]:,}")
                            nstart = ma[j][1] + (s - st)
                            nend = ma[j][1] + (e - st)
                            assert nend - nstart == e - s, f"nstart = {nstart} nend = {nend} e = {e} s = {s}"
                            new_ranges.append((nstart, nend))
                        elif st <= s <= en:
                            if i == len(mappers):
                                print(f"s = {s:,}, e = {e:,}, st = {st:,}, en = {en:,}, destination = {ma[j][1]:,}")
                            nstart = ma[j][1] + (s - st)
                            nend = ma[j][1] + (en - st)
                            assert nend - nstart == en - s, f"nstart = {nstart} nend = {nend} en = {en} s = {s}"
                            new_ranges.append((nstart, nend))
                        elif st <= e <= en:
                            if i == len(mappers):
                                print(f"s = {s:,}, e = {e:,}, st = {st:,}, en = {en:,}, destination = {ma[j][1]:,}")
                            nstart = ma[j][1] + (st - st)
                            nend = ma[j][1] + (e - st)
                            assert nend - nstart == e - st, f"nstart = {nstart} nend = {nend} e = {e} st = {st}"
                            new_ranges.append((nstart, nend))
                ranges = new_ranges
            for s, _ in ranges:
                res = min(res, s)
        print(res)      

if __name__ == '__main__':
    main()

"""
218728775
"""