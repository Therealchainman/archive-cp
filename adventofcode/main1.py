from collections import *
from functools import *
from itertools import *
import sys
import re
import math
import string
import bisect
from parse import compile

# LRL = (MCG, TRC)
# TNJ = (LMV, PMP)
D = compile("{} = ({}, {})")
with open('input.txt', 'r') as f:
    data = f.read().splitlines()
    instructions = data[0]
    n = len(instructions)
    i = 0
    dir = {}
    for line in data[2:]:
        f, L, R = D.parse(line).fixed
        dir[f] = (L, R)
    cnt = 0
    queue = []
    for k in dir.keys():
        if k[-1] == "A":
            queue.append(k)
    while cnt != len(queue):
        nqueue = []
        ncnt = 0
        while queue:
            cur = queue.pop()
            if instructions[i % n] == "L":
                nxt = dir[cur][0]
                nqueue.append(nxt)
                ncnt += nxt[-1] == "Z"
            else:
                nxt = dir[cur][1]
                nqueue.append(nxt)
                ncnt += nxt[-1] == "Z"
        cnt = ncnt
        queue = nqueue
    print(i)