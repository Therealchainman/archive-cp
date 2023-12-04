import os,sys
from io import BytesIO, IOBase
from typing import *
 
# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

# IMPORTS
import itertools
import math
import heapq

# INPUTS
# sys.stdin = open("../tests/01", "r")
# sys.stdin = open("../tests/06", "r")
# sys.stdin = open("../tests/21", "r")
sys.stdin = open("../tests/51", "r")
# sys.stdout = open("output.txt", "w")

# READ IN DATA
N = int(input())
K = int(input())
T = int(input())
R = int(input())

# PARAMETERS
STEP = 0.01 # 80 possible steps from 0 to 4.0

# CONSTANTS
W = 192
init_SINR = [[[[0] * N for _ in range(R)] for _ in range(K)] for _ in range(T)]
interference = [[[[0] * N for _ in range(N)] for _ in range(R)] for _ in range(K)]
# t, n => (sinr, k, r)
sinr_ordered = [[[] for _ in range(N)] for _ in range(T)]
for t, k, r in itertools.product(range(T), range(K), range(R)):
    arr = list(map(float, input().split()))
    for n in range(N):
        init_SINR[t][k][r][n] = arr[n]
        sinr_ordered[t][n].append((arr[n], r, k))
for t, n in itertools.product(range(T), range(N)):
    sinr_ordered[t][n].sort(reverse = True)
for k, r, m in itertools.product(range(K), range(R), range(N)):
    arr = list(map(float, input().split()))
    for n in range(N):
        interference[k][r][n][m] = arr[n]

# CLASSES
class Frame:
    def __init__(self, user_id, TBS, start, end):
        self.user_id = user_id
        self.TBS = TBS
        self.start = start
        self.end = end
    def __repr__(self):
        return f"Frame({self.user_id}, {self.TBS}, {self.start}, {self.end})"

# READ IN FRAMES
J = int(input())
frames = [None] * J
# frames starting at time t
frames_at = [[] for _ in range(T)]
frames_end = [[] for _ in range(T)]
frames_rank = [[] for _ in range(J)]
for j in range(J):
   arr = list(map(int, input().split()))
   TBS, user_id, start, length = arr[1:]
   end = start + length - 1
   frames[j] = Frame(user_id, TBS, start, end)
   frames_at[start].append(j)
   frames_end[end].append(j)

# ARRAYS
cellular_resources = [[R] * K for _ in range(T)]
# t, k, r => (k, j) k = cell, j = frame
visitors = [[None] * R for _ in range(T)]
P = [[[[0] * N for _ in range(R)] for _ in range(K)] for _ in range(T)]
counts = [[[0] * K for _ in range(T)] for _ in range(J)]
product = [[[1] * K for _ in range(T)] for _ in range(J)]
sums = [[[0] * K for _ in range(T)] for _ in range(J)]
# product2 = [[[[1] * N for _ in range(R)] for _ in range(K)] for _ in range(T)]
total_sums = [[0] * T for _ in range(J)]

def trans_helper(sinr):
    return W * math.log2(1 + sinr)

# MATH FUNCTIONS
# geometric mean is the nth root of the product of n numbers
def geometric_mean(pr, n):
    return pow(pr, (1 / n))

# evaluation of the function without interference
def evaluate(product, count):
    return W * count * math.log2(1 + geometric_mean(product, count))

def power_value(index):
    return round(index * STEP, 3)

def binary_search(product, count, neighbor_sums, target):
    left, right = 0, int(4 / STEP)
    while left < right:
        mid = (left + right) >> 1
        total = evaluate(product * power_value(mid), count + 1) + neighbor_sums
        if total <= target:
            left = mid + 1
        else:
            right = mid
    return power_value(left)

# returns resources allocated for one user at a single time step.
# j = frame, t = time, n = user
def allocate(j, t, n, rem):
    for sn, r, k in sinr_ordered[t][n]:
        if cellular_resources[t][k] == 0: continue
        product[j][t][k] *= sn
        take = min(binary_search(product[j][t][k], counts[j][t][k], total_sums[j][t] - sums[j][t][k], rem), cellular_resources[t][k], 4)
        product[j][t][k] *= take
        counts[j][t][k] += 1
        nsum = evaluate(product[j][t][k], counts[j][t][k])
        delta_sum = nsum - sums[j][t][k]
        total_sums[j][t] += delta_sum
        sums[j][t][k] = nsum
        P[t][k][r][n] = take
        cellular_resources[t][k] -= take
        visitors[t][r] = n
        if total_sums[j][t] >= rem: break
    return sum(sums[j][t])

def dry_allocate(t, n):
    data = 0
    for k in range(K): # iterate over cells
        sinr = []
        for r in range(R): # iterate over resources
            sinr.append((init_SINR[t][k][r][n], r))
        sinr.sort(reverse = True)
        to_give = R
        product = 1
        cnt = 0
        for i in range(R):
            sn, r = sinr[i]
            if to_give == 0: break
            product *= sn
            take = min(to_give, 4)
            to_give -= take
            cnt += 1
            product *= sn * take
        gm = pow(product, (1 / cnt))
        for _ in range(cnt):
            data += trans_helper(gm)
    return data

def process_frames():
    frames_suffix_sum = [[0] * (T + 1) for _ in range(J)]
    frame_windows = set()
    for t in reversed(range(T)):
        for j in frames_end[t]:
            frame_windows.add(j)
        for j in frame_windows:
            # rough estimate of best it can calculate
            # can I do better here? 
            d = dry_allocate(t, frames[j].user_id)
            frames_rank[j].append((d, t))
            frames_suffix_sum[j][t] = frames_suffix_sum[j][t + 1] + d
        for j in frames_at[t]:
            frames_rank[j].sort(reverse = True)
            frame_windows.discard(j)
    return frames_suffix_sum

frame_resources = [0] * J

frames_suffix_sum = process_frames()
frame_heap = []

data_needs = sorted([(frames[j].TBS,  j) for j in range(J)], reverse = True)

# FIRST ALLOCATION OF RESOURCES
frame_allocation = [None] * T
resources_allocated = [0] * T
before = [0] * T
after = [0] * T
allocated = [0] * J
end_times = set()
remaining = [frames[j].TBS for j in range(J)]
power_sum = 0
for needed, j in data_needs:
    rem = needed
    days = end_time = 0
    times = []
    for d, t in frames_rank[j]:
        if frame_allocation[t] is None:
            times.append(t)
            days += 1
            frame_allocation[t] = j 
            resources_allocated[t] = allocate(j, t, frames[j].user_id, rem)
            before[t] = rem
            rem -= resources_allocated[t]
            after[t] = rem
            end_time = max(end_time, t)
        if rem <= 0: 
            end_times.add(end_time)
            allocated[j] = 1
            break
    if rem > 0:
        for t in times:
            frame_allocation[t] = None
            for k, r in itertools.product(range(K), range(R)):
                P[t][k][r][frames[j].user_id] = 0
                cellular_resources[t][k] = R
                visitors[t][r] = None
            resources_allocated[t] = 0
            before[t] = 0
            after[t] = 0

def transfer(t):
    for r in range(R):
        if visitors[t][r] is not None: continue
        while frame_heap:
            rem, delta, j = heapq.heappop(frame_heap)
            if t > frames[j].end or rem > delta:  # expired
                continue
            n = frames[j].user_id
            freq_sinr = []
            for k in range(K):
                freq_sinr.append((init_SINR[t][k][r][n], k))
            freq_sinr.sort(reverse = True)
            for i in range(K):
                sn, k = freq_sinr[i]
                if cellular_resources[t][k] == 0: continue
                product[j][t][k] *= sn
                take = min(binary_search(product[j][t][k], counts[j][t][k], total_sums[j][t] - sums[j][t][k], rem), cellular_resources[t][k], 4)
                product[j][t][k] *= take
                counts[j][t][k] += 1
                nsum = evaluate(product[j][t][k], counts[j][t][k])
                delta_sum = nsum - sums[j][t][k]
                total_sums[j][t] += delta_sum
                sums[j][t][k] = nsum
                P[t][k][r][n] = take
                cellular_resources[t][k] -= take
                if total_sums[j][t] >= rem: break
            visitors[t][r] = n
            if total_sums[j][t] >= frames[j].TBS:
                allocated[j] = 1
            else:
                heapq.heappush(frame_heap, (frames[j].TBS - total_sums[j][t], frames_suffix_sum[j][t] - (frames[j].TBS - total_sums[j][t]), j))
            break

# SECOND ALLOCATION OF RESOURCES
for t in range(T):
    for j in frames_at[t]:
        if allocated[j] == 1: continue
        heapq.heappush(frame_heap, (frames[j].TBS, frames_suffix_sum[j][t] - frames[j].TBS, j))
    transfer(t)

print("num_frames", sum(allocated))
print("target frames: ", J)

# for t, k, r in itertools.product(range(T), range(K), range(R)):
#     print(*P[t][k][r])
