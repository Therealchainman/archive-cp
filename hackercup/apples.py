import sys

# name = "two_apples_a_day_sample_input.txt"
# name = "two_apples_a_day_validation_input.txt"
name = "two_apples_a_day_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

import math
from collections import Counter

def main(t):
    N = int(input())
    M = 2 * N - 1
    arr = list(map(int, input().split()))
    res = math.inf
    if N == 1:
        res = arr[0]
        return print(f"Case #{t}: {res}")
    arr.sort()
    counts = Counter()
    for i in range(1, N):
        s = arr[i] + arr[M - i]
        counts[s] += 1
    if len(counts) == 1:
        v = list(counts).pop()
        if v > arr[0]: res = min(res, v - arr[0])
    for i in range(1, M):
        x = arr[M - i - (0 if i < N else 1)]
        ps = arr[i] + x
        s = arr[i - 1] + x
        counts[ps] -= 1
        counts[s] += 1
        if counts[ps] == 0: counts.pop(ps)
        if (len(counts) == 1):
            v = list(counts).pop()
            if v > arr[i]: res = min(res, v - arr[i])
    if res == math.inf: res = -1
    print(f"Case #{t}: {res}")

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        main(t)