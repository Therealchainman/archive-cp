# Google Kickstart 2022 Round C

## Summary

## New Password

### Solution 1: string

```py
import string
SPECIAL_CHARACTERS = '#@*&'
string.digits
string.ascii_lowercase
def main():
    num_digits = num_lower = num_upper = num_special = 0
    N = int(input())
    S = input()
    for ch in S:
        num_digits += (ch in string.digits)
        num_lower += (ch in string.ascii_lowercase)
        num_upper += (ch in string.ascii_uppercase)
        num_special += (ch in SPECIAL_CHARACTERS)
    num_missing = 7 - N
    to_add = []
    if num_digits == 0:
        to_add.append('1')
        num_missing -= 1
    if num_lower == 0:
        to_add.append('a')
        num_missing -= 1
    if num_upper == 0:
        to_add.append('A')
        num_missing -= 1
    if num_special == 0:
        to_add.append('#')
        num_missing -= 1
    while num_missing > 0:
        num_missing -= 1
        to_add.append('1')
    return S + "".join(to_add)
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## Range Partition

### Solution 1: math + greedy

```py
import sys
sys.setrecursionlimit(1000000)
POS = "POSSIBLE"
IMP = "IMPOSSIBLE"
def main():
    N, X, Y = map(int,input().split())
    sum_N = N*(N+1)//2
    if sum_N%(X+Y)!=0: return IMP
    partition_sum = (sum_N//(X+Y))*X
    arr = []
    def partition(N, partition_sum):
        if N == 0 or partition_sum == 0: return
        if N > partition_sum:
            partition(N-1, partition_sum)
        else:
            arr.append(N)
            partition(N-1, partition_sum-N)
    partition(N,partition_sum)
    return POS + '\n' f'{len(arr)}' '\n' + ' '.join(map(str, arr))
            

if __name__ == '__main__':
    T = int(input())
    for t in range(1,T+1):
        print(f"Case #{t}: {main()}")
```

## 

###

##

###
