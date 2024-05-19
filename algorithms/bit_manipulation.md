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

## lowbit function

This function returns the lowest bit that is set in a given integer.  It does so by using the bitwise AND operator with the two's complement of the input integer.  This is a common operation in bit manipulation and is used in many algorithms that require bit manipulation.

```py
def lowbit(x):
    return x & -x
```


## Important fact about binary representations

1000 > 0111, This is useful for solving many problems, because now you know you just need to set the highest bit to 1 and you will get the maximum number.

## Bit Equations

Bit equation that relates addition to bitwise XOR and AND operations.  This is useful for solving problems that require bit manipulation.  This is a useful way to convert from one to another, if it makes something simpler.
x+y=(x⊕y)+2⋅(x&y)