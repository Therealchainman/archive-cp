# Codeforces Round 778

## Summary

Overall it was an interesting contest, I solved the first 3 questions in less than an hour.  Mostly I spent too much time with some small mistakes.  But the 4th question was at my limit of knowledge. I worked on some system of linear equations and using LCM, but I just don't know how to write the code for it.  I can solve by hand if I'm given equation.  I was able to observe that it was a tree.  And so from the hints I see you can use a dfs traversal of the tree, and use the prime sieve of Erastothenes algorithm to compute.  Still need to learn how to do that. 


### Maximum Cake Tastiness

```py
T = int(input())
for _ in range(T):
    N = int(input())
    arr = list(map(int,input().split()))
    arr.sort()
    print(arr[-1]+arr[-2])
```


### Prefix Removals

Can use Counter as well, would be simpler code actually

```py
T = int(input())
for _ in range(T):
    S = input()
    counts = [0]*26
    for ch in S:
        counts[ord(ch)-ord('a')] += 1
    i = 0
    while counts[ord(S[i])-ord('a')] > 1:
        counts[ord(S[i])-ord('a')] -= 1
        i += 1
    print(S[i:])
```


### Alice and the Cake

I thought a heap should be optimal enough for this problem.  

Using the fact that dictionaries are in order of addition from python 3.6+ or so.

So sorted pieces then add them to dictionary in order

```py
from heapq import heappop, heappush, heapify
T = int(input())
for _ in range(T):
    N = int(input())
    pieces = sorted(list(map(int,input().split())))
    def floor(x):
        return x//2
    def ceil(x):
        return (x+1)//2
    def can_cut():
        sum_ = 0
        counts = {}
        for x in pieces:
            cnt = counts.setdefault(x, 0)
            counts.update({x: cnt+1})
            sum_ += x
        minheap = [sum_]
        while minheap:
            elem = heappop(minheap)
            first_key = next(iter(counts.items()))[0]
            if elem < first_key:
                return False
            if elem in counts:
                counts[elem]-=1
                if counts[elem]==0:
                    del counts[elem]
                continue
            x, y = floor(elem), ceil(elem)
            heappush(minheap, x)
            heappush(minheap, y)
        return True

    if can_cut():
        print('YES')
    else:
        print('NO')
```


###

I couldn't solve this one, going to try to practice it later today when I get chance, I know it is dfs and prime sieve related


This is the best solution I saw.  

```py
import os,sys
from io import BytesIO, IOBase
 
def main():
    mod = 998244353
    lim = 200001
    sieve,primes = [[] for _ in range(lim)],[]
    for i in range(2,lim):
        if not len(sieve[i]):
            primes.append(i)
            for j in range(i,lim,i):
                xx = j
                while not xx%i:
                    sieve[j].append(i)
                    xx //= i
    for _ in range(int(input())):
        n = int(input())
        path = [[] for _ in range(n)]
        col = {}
        for _ in range(n-1):
            i,j,x,y = map(int,input().split())
            path[i-1].append((j-1,x,y))
            path[j-1].append((i-1,y,x))
            col[(i-1,j-1)] = (x,y)
            col[(j-1,i-1)] = (y,x)
        curr = {i:0 for i in primes if i<=n}
        mini = [0]*(n+1)
        st,poi,visi = [0],[0]*n,[1]+[0]*(n-1)
        while len(st):
            x,y = st[-1],path[st[-1]]
            if poi[x] != len(y) and visi[y[poi[x]][0]]:
                poi[x] += 1
            if poi[x] == len(y):
                st.pop()
                if len(st):
                    co,co1 = col[(x,st[-1])]
                    for xx in sieve[co]:
                        curr[xx] -= 1
                    for xx in sieve[co1]:
                        curr[xx] += 1
                    for xx in sieve[co]+sieve[co1]:
                        mini[xx] = min(mini[xx],curr[xx])
            else:
                z,co,co1 = y[poi[x]]
                visi[z] = 1
                st.append(z)
                poi[x] += 1
                for xx in sieve[co]:
                    curr[xx] -= 1
                for xx in sieve[co1]:
                    curr[xx] += 1
                for xx in sieve[co]+sieve[co1]:
                    mini[xx] = min(mini[xx],curr[xx])
        ini = 1
        for i in primes:
            if i >= len(mini):
                break
            ini = (ini*pow(i,-mini[i],mod))%mod
        st,poi,visi = [0],[0]*n,[ini]+[0]*(n-1)
        while len(st):
            x,y = st[-1],path[st[-1]]
            if poi[x] != len(y) and visi[y[poi[x]][0]]:
                poi[x] += 1
            if poi[x] == len(y):
                st.pop()
            else:
                z,co,co1 = y[poi[x]]
                visi[z] = (visi[x]*pow(co,mod-2,mod)*co1)%mod
                st.append(z)
                poi[x] += 1
        ans = 0
        for i in visi:
            ans = (ans+i)%mod
        print(ans)
 
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
if __name__ == "__main__":
    main()
```

```cpp
#include <bits/stdc++.h>
using namespace std;
#define rep(i, a, b) for (int i = (a); i < (b); i++)
struct Edge { int to, num, den; };
using Graph = vector<vector<Edge> >;
using ll = long long;
const int maxn=2e5+5;
vector<int> pf(maxn);
const ll mod = 998244353;
vector<ll> inv(maxn);
 
void dfs(int x, Graph &g, vector<bool> &vis, vector<int> &ex, vector<int> &mn_ex, ll &num, ll &sum) {
    sum = (sum + num) % mod;
    vis[x] = true;
    for(Edge e: g[x]) if(!vis[e.to]) {
        for(int k = e.num; k!=1; k/=pf[k]) ex[pf[k]]++, num = num * pf[k] % mod;
        for(int k = e.den; k!=1; k/=pf[k]) ex[pf[k]]--, num = num * inv[pf[k]] % mod, mn_ex[pf[k]] = min(mn_ex[pf[k]], ex[pf[k]]);
        dfs(e.to, g, vis, ex, mn_ex, num, sum);
        for(int k = e.num; k!=1; k/=pf[k]) ex[pf[k]]--, num = num * inv[pf[k]] % mod;
        for(int k = e.den; k!=1; k/=pf[k]) ex[pf[k]]++, num = num * pf[k] % mod;
    }
}
 
int main() {
    ios_base::sync_with_stdio(0);
 
    rep(i,2,maxn) pf[i] = i;
    rep(i,2,maxn) if(pf[i] == i) for(int j=i; j < maxn; j+=i) pf[j] = i;
 
    inv[1] = 1;
    rep(i,2,maxn) inv[i] = mod - mod / i * inv[mod % i] % mod;
 
    int TC; cin>>TC;
    while(TC--) {
        int N; cin>>N;
        Graph g(N+1);
        rep(i,0,N-1) {
            int a,b,x,y;
            cin>>a>>b>>x>>y;
            g[a].push_back({b,y,x});
            g[b].push_back({a,x,y});
        }
        ll sum = 0, num = 1;
        vector<int> ex(N+1), mn_ex(N+1);
        vector<bool> vis(N+1);
        dfs(1, g, vis, ex, mn_ex, num, sum);
        rep(i,2,N+1) while(mn_ex[i] < 0) mn_ex[i]++, sum = sum * i % mod;
 
        cout<<sum<<"\n";
    }
}
```