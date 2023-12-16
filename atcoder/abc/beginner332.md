# Atcoder Beginner Contest 332

## D - Swapping Puzzle 

### Solution 1:  enumerate permutations, selection sort

```py
from itertools import permutations, product
import math

def main():
    R, C = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(R)]
    B = [list(map(int, input().split())) for _ in range(R)]
    res = math.inf
    def check(rows, cols):
        for (i, r), (j, c) in product(enumerate(rows), enumerate(cols)):
            if A[i][j] != B[r][c]:
                return False
        return True
    def swaps(N, arr):
        res = 0
        vals = list(range(N))
        for v in vals:
            for i, x in enumerate(arr):
                if x == v: break
            while x < i:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                res += 1
                i -= 1
        return res
    for rows, cols in product(permutations(range(R)), permutations(range(C))):
        if check(rows, cols):
            res = min(res, swaps(R, list(rows)) + swaps(C, list(cols)))
    res = res if res != math.inf else -1
    print(res)
    
if __name__ == '__main__':
    main()
```

## E - Lucky bag 

### Solution 1:  dynamic programming with bitmasks, enumerating submasks

```py
import math

def main():
    N, D = map(int, input().split())
    arr = list(map(int, input().split()))
    avg = sum(arr) / D
    dp = [[math.inf] * (1 << N) for _ in range(D + 1)]
    def bag(mask):
        weight = sum(arr[i] for i in range(N) if (mask >> i) & 1)
        return (weight - avg) ** 2
    # base case is for every possible set of items in one bag dp[1][mask]
    for mask in range(1 << N): 
        dp[1][mask] = bag(mask)
    for i in range(2, D + 1): # i bags
        for mask in range(1, 1 << N): # set of items taken in i bags
            submask = mask
            dp[i][mask] = dp[i - 1][mask] + dp[1][0] # take no items into new bag
            while submask > 0:
                submask = (submask - 1) & mask
                dp[i][mask] = min(dp[i][mask], dp[i - 1][submask] + dp[1][mask ^ submask])
    ans = dp[-1][-1] / D
    print(ans)

if __name__ == '__main__':
    main()
```

## F - Random Update Query 

### Solution 1: lazy segment tree

```py
class lazy_segtree():
    def update(self,k):self.d[k]=self.op(self.d[2*k],self.d[2*k+1])
    def all_apply(self,k,f):
        self.d[k]=self.mapping(f,self.d[k])
        if (k<self.size):self.lz[k]=self.composition(f,self.lz[k])
    def push(self,k):
        self.all_apply(2*k,self.lz[k])
        self.all_apply(2*k+1,self.lz[k])
        self.lz[k]=self.identity
    def __init__(self,V,OP,E,MAPPING,COMPOSITION,ID):
        self.n=len(V)
        self.log=(self.n-1).bit_length()
        self.size=1<<self.log
        self.d=[E for i in range(2*self.size)]
        self.lz=[ID for i in range(self.size)]
        self.e=E
        self.op=OP
        self.mapping=MAPPING
        self.composition=COMPOSITION
        self.identity=ID
        for i in range(self.n):self.d[self.size+i]=V[i]
        for i in range(self.size-1,0,-1):self.update(i)
    def set(self,p,x):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self.push(p>>i)
        self.d[p]=x
        for i in range(1,self.log+1):self.update(p>>i)
    def get(self,p):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self.push(p>>i)
        return self.d[p]
    def prod(self,l,r):
        assert 0<=l and l<=r and r<=self.n
        if l==r:return self.e
        l+=self.size
        r+=self.size
        for i in range(self.log,0,-1):
            if (((l>>i)<<i)!=l):self.push(l>>i)
            if (((r>>i)<<i)!=r):self.push(r>>i)
        sml,smr=self.e,self.e
        while(l<r):
            if l&1:
                sml=self.op(sml,self.d[l])
                l+=1
            if r&1:
                r-=1
                smr=self.op(self.d[r],smr)
            l>>=1
            r>>=1
        return self.op(sml,smr)
    def all_prod(self):return self.d[1]
    def apply_point(self,p,f):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self.push(p>>i)
        self.d[p]=self.mapping(f,self.d[p])
        for i in range(1,self.log+1):self.update(p>>i)
    def apply(self,l,r,f):
        assert 0<=l and l<=r and r<=self.n
        if l==r:return
        l+=self.size
        r+=self.size
        for i in range(self.log,0,-1):
            if (((l>>i)<<i)!=l):self.push(l>>i)
            if (((r>>i)<<i)!=r):self.push((r-1)>>i)
        l2,r2=l,r
        while(l<r):
            if (l&1):
                self.all_apply(l,f)
                l+=1
            if (r&1):
                r-=1
                self.all_apply(r,f)
            l>>=1
            r>>=1
        l,r=l2,r2
        for i in range(1,self.log+1):
            if (((l>>i)<<i)!=l):self.update(l>>i)
            if (((r>>i)<<i)!=r):self.update((r-1)>>i)
    def max_right(self,l,g):
        assert 0<=l and l<=self.n
        assert g(self.e)
        if l==self.n:return self.n
        l+=self.size
        for i in range(self.log,0,-1):self.push(l>>i)
        sm=self.e
        while(1):
            while(l%2==0):l>>=1
            if not(g(self.op(sm,self.d[l]))):
                while(l<self.size):
                    self.push(l)
                    l=(2*l)
                    if (g(self.op(sm,self.d[l]))):
                        sm=self.op(sm,self.d[l])
                        l+=1
                return l-self.size
            sm=self.op(sm,self.d[l])
            l+=1
            if (l&-l)==l:break
        return self.n
    def min_left(self,r,g):
        assert (0<=r and r<=self.n)
        assert g(self.e)
        if r==0:return 0
        r+=self.size
        for i in range(self.log,0,-1):self.push((r-1)>>i)
        sm=self.e
        while(1):
            r-=1
            while(r>1 and (r%2)):r>>=1
            if not(g(self.op(self.d[r],sm))):
                while(r<self.size):
                    self.push(r)
                    r=(2*r+1)
                    if g(self.op(self.d[r],sm)):
                        sm=self.op(self.d[r],sm)
                        r-=1
                return r+1-self.size
            sm=self.op(self.d[r],sm)
            if (r&-r)==r:break
        return 0

mod = 998244353

# I don't get this? 
def op(a, b):
    return ((a[0] + b[0]) % mod, (a[1] + b[1]) % mod)
# maps F, S to S
def mapping(f, x):
    return ((f[0] * x[0] + x[1] * f[1]) % mod, x[1])
# composition of F, F to F
def composition(f, g):
    return ((f[0] * g[0]) % mod, (g[1] * f[0] + f[1]) % mod)
# identity element for op (0, 0)
# identity element for mapping (1, 0)

def mod_inverse(v):
    return pow(v, mod - 2, mod)

def main():
    N, M = map(int, input().split())
    arr = list(map(int, input().split()))
    seg = lazy_segtree([(x, 1) for x in arr], op, (0, 0), mapping, composition, (1, 0))
    for _ in range(M):
        left, right, x = map(int, input().split())
        left -= 1
        delta = right - left
        a = (1 - mod_inverse(delta)) % mod
        b = (x * mod_inverse(delta)) % mod
        seg.apply(left, right, (a, b))
    print(*[seg.get(i)[0] for i in range(N)])

if __name__ == '__main__':
    main()
```

## G - Not Too Many Balls 

### Solution 1:  max flow min cut theorem and dynamic programming

```py
from collections import defaultdict

def main():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    max_k = N * (N + 1) // 2
    INF = 1 << 60
    dp = [INF] * (max_k + 1)
    dp[0] = 0
    for i, a in enumerate(A, start = 1):
        available_k = i * (i - 1) // 2
        for k in range(available_k, -1, -1):
            dp[k + i] = min(dp[k + i], dp[k] + a)
    over_bs = defaultdict(list)
    for j, b in enumerate(B, start = 1):
        max_k2 = b // j
        over_bs[max_k2].append(j)
    j_sum = M * (M + 1) // 2
    b_sum = 0
    ans = INF
    for k2 in range(max_k + 1):
        ans = min(ans, dp[max_k - k2] + k2 * j_sum + b_sum)
        if k2 in over_bs:
            for j in over_bs[k2]:
                j_sum -= j
                b_sum += B[j - 1]
    print(ans)

if __name__ == '__main__':
    main()
```
