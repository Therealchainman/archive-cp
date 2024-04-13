# Bit Manipulation


## Counting the number of integers within range with bit b set

This shows it being used for each bit j, and given intervals at each index i.  And it calculate how many integers in the range bit j is set. 

Calculates the number of integers within the range 0 <= x <= N, (where N is represented by the paramter x) that have the bth bit set.  It does so by leveraging the periodic nature of binary numbers and the specific bit patterns within a given range. 


```py
def count_bits(x, b):
    cycle_len = 2 ** (b + 1)
    cycle_cnt = x // cycle_len
    cycle_rem = x % cycle_len - 2 ** b + 1
    return cycle_cnt * 2 ** b + max(0, cycle_rem)

bfreq = [[0] * BITS for _ in range(N)]
for i in range(N):
    for j in range(BITS):
        bfreq[i][j] = count_bits(R[i], j) % MOD
        if L[i] > 0: bfreq[i][j] = (bfreq[i][j] - count_bits(L[i] - 1, j)) % MOD
```