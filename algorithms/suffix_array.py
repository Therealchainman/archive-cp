"""
Suffix array is an array of integers, where the integers represent the suffix from a string.
the integer in suffix array represents the starting index for the suffix. 
suffix array is these suffix index sorted in order of suffix order from ascending order

sorting is O(n+k) where k is the range of values in the string.

"""
from typing import List

def radix_sort(p: List[int], c: List[int]) -> List[int]:
    n = len(p)
    cnt = [0]*n
    next_p = [0]*n
    for cls_ in c:
        cnt[cls_] += 1
    pos = [0]*n
    for i in range(1,n):
        pos[i] = pos[i-1] + cnt[i-1]
    for pi in p:
        cls_i = c[pi]
        next_p[pos[cls_i]] = pi
        pos[cls_i] += 1
    return next_p


def suffix_array(s: str) -> str:
    n = len(s)
    p, c = [0]*n, [0]*n
    arr = [None]*n
    for i, ch in enumerate(s):
        arr[i] = (ch, i)
    arr.sort()
    for i, (_, j) in enumerate(arr):
        p[i] = j
    c[p[0]] = 0
    for i in range(1,n):
        c[p[i]] = c[p[i-1]] + (arr[i][0] != arr[i-1][0])
    k = 1
    is_finished = False
    while k < n and not is_finished:
        for i in range(n):
            p[i] = (p[i] - k + n)%n
        p = radix_sort(p, c)
        next_c = [0]*n
        next_c[p[0]] = 0
        is_finished = True
        for i in range(1,n):
            prev_segments = (c[p[i-1]], c[(p[i-1]+k)%n])
            current_segments = (c[p[i]], c[(p[i]+k)%n])
            next_c[p[i]] = next_c[p[i-1]] + (prev_segments != current_segments)
            is_finished &= (next_c[p[i]] != next_c[p[i-1]])
        k <<= 1
        c = next_c
    return ' '.join(map(str, p))