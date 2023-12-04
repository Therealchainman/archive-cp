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
from itertools import product
from collections import defaultdict, Counter
import random
import math
import time
import heapq
import bisect

# sys.stdin = open("tests/01", "r")
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
for t, k, r in product(range(T), range(K), range(R)):
    arr = list(map(float, input().split()))
    for n in range(N):
        init_SINR[t][k][r][n] = arr[n]
        sinr_ordered[t][n].append((arr[n], k, r))
for t, n in product(range(T), range(N)):
    sinr_ordered[t][n].sort(reverse = True)
for k, r, m in product(range(K), range(R), range(N)):
    arr = list(map(float, input().split()))
    for n in range(N):
        interference[k][r][n][m] = arr[n]

class Frame:
    def __init__(self, user_id, TBS, start, end):
        self.user_id = user_id
        self.TBS = TBS
        self.start = start
        self.end = end
    def __repr__(self):
        return f"Frame({self.user_id}, {self.TBS}, {self.start}, {self.end})"

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

P = [[[[0] * N for _ in range(R)] for _ in range(K)] for _ in range(T)]

def trans_helper(sinr):
    return W * math.log2(1 + sinr)

def allocate_resources(t, n, track = False):
    ps = data = 0
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
            take = min(to_give, 4) # TODO: Equation to calculate how much to take
            if track:
                P[t][k][r][n] = take
            to_give -= take
            cnt += 1
            product *= sn * take
            ps += take
        gm = pow(product, (1 / cnt))
        for _ in range(cnt):
            data += trans_helper(gm)
    return data, ps


def process_frames():
    frames_suffix_sum = [[0] * (T + 1) for _ in range(J)]
    frame_windows = set()
    for t in reversed(range(T)):
        for j in frames_end[t]:
            frame_windows.add(j)
        for j in frame_windows:
            # rough estimate of best it can calculate
            # can I do better here? 
            d, _ = allocate_resources(t, frames[j].user_id)
            frames_rank[j].append((d, t))
            frames_suffix_sum[j][t] = frames_suffix_sum[j][t + 1] + d
        for j in frames_at[t]:
            frames_rank[j].sort(reverse = True)
            frame_windows.discard(j)
    return frames_suffix_sum

# geometric mean is the nth root of the product of n numbers
def geometric_mean(pr, n):
    return pow(pr, (1 / n))

# evaluating over a single r, but over multiple k? 
def shannon_capacity(x):
    return W * math.log2(1 + x)

# just an approximation
def calc(cur, C, x):
    return W * C * math.log2(1 + geometric_mean(cur * x, C))

def power_value(index):
    return round(index * STEP, 3)

# pass in cur which is the surrent snr, so power_value is the power to multiply by
def binary_search(cur, target):
    left, right = 0, int(4 / STEP)
    while left < right:
        mid = (left + right) >> 1
        if shannon_capacity(cur * power_value(mid)) <= target:
            left = mid + 1
        else:
            right = mid
    return power_value(left)
frame_resources = [0] * J
def multi_transmit_data(t, designated_frame):
    global num_frames
    ps = 0
    leftover = [R] * K
    for r in range(R):
        if designated_frame is not None and remaining[designated_frame] > 0:
            n = frames[designated_frame].user_id
            freq_sinr = []
            for k in range(K):
                freq_sinr.append((init_SINR[t][k][r][n], k))
            freq_sinr.sort(reverse = True)
            for i in range(K):
                sn, k = freq_sinr[i]
                if leftover[k] == 0: continue
                if remaining[designated_frame] <= 0: break # probably finished
                take = min(binary_search(sn, remaining[designated_frame]), leftover[k])
                frame_resources[designated_frame] += take
                P[t][k][r][n] = take
                leftover[k] -= take
                remaining[designated_frame] -= shannon_capacity(take * sn)
                ps += take
            num_frames += remaining[designated_frame] <= 0
        else:
            while frame_heap:
                rem, delta, idx = heapq.heappop(frame_heap)
                if t > frames[idx].end or rem > delta:  # expired
                    lost_frames.add(idx)
                    continue
                n = frames[idx].user_id
                freq_sinr = []
                for k in range(K):
                    freq_sinr.append((init_SINR[t][k][r][n], k))
                freq_sinr.sort(reverse = True)
                for i in range(K):
                    sn, k = freq_sinr[i]
                    if leftover[k] == 0: continue
                    if remaining[idx] <= 0: break # probably finished
                    take = min(binary_search(sn, remaining[idx]), leftover[k])
                    frame_resources[idx] += take
                    P[t][k][r][n] = take
                    leftover[k] -= take
                    remaining[idx] -= shannon_capacity(take * sn)
                    ps += take
                num_frames += remaining[idx] <= 0
                if remaining[idx] > 0:
                    heapq.heappush(frame_heap, (remaining[idx], frames_suffix_sum[idx][t + 1] - remaining[idx], idx))
                break
    return ps

frames_suffix_sum = process_frames()
frame_heap = []

lost_frames = set()
num_frames = power_sum = 0

data_needs = sorted([(frames[j].TBS,  j) for j in range(J)], reverse = True)

frame_allocation = [None] * T
resources_allocated = [0] * T
before = [None] * T
after = [None] * T
allocated = set()
remaining = [frames[j].TBS for j in range(J)]
i = J // 3
for index in range(J):
    needed, j = data_needs[i % J]
    rem = needed
    days = 0
    times = []
    for d, t in frames_rank[j]:
        if frame_allocation[t] is None:
            times.append(t)
            days += 1
            frame_allocation[t] = j 
            resources_allocated[t] = d
            before[t] = rem
            rem -= d
            after[t] = rem
        if rem <= 0: 
            allocated.add(j)
            break
    if rem > 0:
        for t in times:
            frame_allocation[t] = None
            resources_allocated[t] = 0
            before[t] = None
            after[t] = None
    i += 1
power_sum = 0
# for t in range(T):
#     # if t % 100 == 0: print("t", t, flush = True)
#     for j in frames_at[t]:
#         heapq.heappush(frame_heap, (frames[j].TBS, frames_suffix_sum[j][t] - frames[j].TBS, j))
#         # heapq.heappush(frame_heap, (frames_suffix_sum[j][t] - frames[j].TBS, frames[j].TBS, j))
#     # assign all the k, r pairs
#     power_sum += multi_transmit_data(t)
for t in range(T):
    for j in frames_at[t]:
        if j in allocated: continue
        heapq.heappush(frame_heap, (frames[j].TBS, frames_suffix_sum[j][t] - frames[j].TBS, j))
    power_sum += multi_transmit_data(t, frame_allocation[t])

# times = set()
# for j in range(J):
#     for t in range(frames[j].start, frames[j].end + 1):
#         times.add(t)
# times = sorted(times)
# for t in range(T):
#     if frame_allocation[t] is None:
#         print("t", t)
#         for j in set(range(J)) - completed:
#             if frames[j].start <= t <= frames[j].end:
#                 print("j", j, frames[j])

score = num_frames - pow(10, -6) * power_sum
print("num_frames", num_frames, "power_sum", power_sum, "score", score)
print("target frames: ", J)

# for t, k, r in product(range(T), range(K), range(R)):
#     print(*P[t][k][r])

# print("completed", len(completed), completed)
# print(frame_allocation)

# for j in range(J):
#     mn, mx = math.inf, -math.inf
#     for d, t in frames_rank[j]:
#         mn = min(mn, d)
#         mx = max(mx, d)
#     print("j", j, "mn", mn, "mx", mx)
# # print("R", R, "K", K, "T", T)
# for t in range(T):
#     # if t % 100 == 0: print("t", t, flush = True)
#     for j in frames_at[t]:
#         heapq.heappush(frame_heap, (frames[j].TBS, frames_suffix_sum[j][t] - frames[j].TBS, j))
#         # heapq.heappush(frame_heap, (frames_suffix_sum[j][t] - frames[j].TBS, frames[j].TBS, j))
#     # assign all the k, r pairs
#     power_sum += multi_transmit_data(t)

# print("lost frames", lost_frames)

