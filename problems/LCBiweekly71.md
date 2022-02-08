# Leetcode Biweekly Contest 71




## 2165. Smallest Value of the Rearranged Number

### Solution:


```py
class Solution:
    def smallestNumber(self, num: int) -> int:
        if num==0: return num
        sign = False if num<0 else True
        smallest = math.inf
        if sign:
            num = list(str(num))
            num.sort()
            cntZeros = sum(1 for x in num if x=='0')
            return int("".join(num[cntZeros:cntZeros+1] + num[:cntZeros]+num[cntZeros+1:]))
        num = list(str(num)[1:])
        num.sort(reverse=True)
        return -int("".join(num))
```

## 2166. Bitset Design

### Solution: datastructure impelementation with lists of 0 and 1s, storing bits and flipped bits, and needed variables

```py
class Bitset:

    def __init__(self, size: int):
        self.bits = [0 for _ in range(size)]
        self.fbits = [1 for _ in range(size)]
        self.cnt_ones = 0
        self.cnt_ones_flipped = size
        self.size = size

    def fix(self, idx: int) -> None:
        if self.bits[idx]==0:
            self.bits[idx]=1
            self.fbits[idx]=0
            self.cnt_ones += 1
            self.cnt_ones_flipped-=1

    def unfix(self, idx: int) -> None:
        if self.bits[idx]==1:
            self.bits[idx]=0
            self.fbits[idx]=1
            self.cnt_ones-=1
            self.cnt_ones_flipped+=1

    def flip(self) -> None:
        self.bits, self.fbits = self.fbits, self.bits
        self.cnt_ones, self.cnt_ones_flipped = self.cnt_ones_flipped, self.cnt_ones

    def all(self) -> bool:
        return self.cnt_ones==self.size

    def one(self) -> bool:
        return self.cnt_ones>0

    def count(self) -> int:
        return self.cnt_ones

    def toString(self) -> str:
        return "".join(map(str,self.bits))
```