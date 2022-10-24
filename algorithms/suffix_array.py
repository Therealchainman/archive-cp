"""
Suffix array is an array of integers, where the integers represent the suffix from a string.
the integer in suffix array represents the starting index for the suffix. 
suffix array is these suffix index sorted in order of suffix order from ascending order

"""
from typing import List
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
            arr[i] = (c[i], c[(i+k)%n], i)
        arr.sort()
        for i, (_, _, j) in enumerate(arr):
            p[i] = j
        c[p[0]] = 0
        is_finished = True
        for i in range(1,n):
            c[p[i]] = c[p[i-1]] + (arr[i][:2] != arr[i-1][:2])
            is_finished &= (c[p[i]] != c[p[i-1]])
        k <<= 1
    return ' '.join(map(str, p))