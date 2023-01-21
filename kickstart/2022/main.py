
import math
from collections import deque
class UnionFind:
    def __init__(self):
        self.size = dict()
        self.parent = dict()
    
    def find(self,i: int) -> int:
        if i not in self.parent:
            self.size[i] = 1
            self.parent[i] = i
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in self.parent)

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
        
def main():
    N = int(input())
    arr = list(map(int, input().split()))
    dsu = UnionFind()
    for i, num in enumerate(arr, start = 1):
        dsu.union(i, num)
    cycleSizes = [0]*(N+1)
    for i in range(1, N+1):
        # i is a representative (root) node for a connected component
        if i == dsu.find(i):
            cycleSizes[dsu.size[i]] += 1
    # bounded knapsack problem
    dp = [math.inf]*(N+1)
    dp[0] = 0
    for cycle_len in range(1, N + 1):
        cnt = cycleSizes[cycle_len]
        if cnt == 0: continue
        # simulates adding to existing solutions
        # this will be ran approximatley sqrt(N) times
        # sliding window for each gap
        for i in range(N, N - cycle_len, -1):
            min_window = deque()
            for right in range(i, -1, -cycle_len):
                left = right - cnt*cycle_len
                if min_window and min_window[0][1] >= right:
                    min_window.popleft()
                while min_window and dp[left] + cnt <= min_window[-1][0] + (right - min_window[-1][1])//cycle_len:
                    min_window.pop()
                min_window.append((dp[left], left))
                dp[right] = min(dp[right], min_window[0][0] + (right - min_window[0][1])//cycle_len)
    # simulates breaking, can always perform minimum swaps and then you can always break to get something smaller so that requires 1 extra move
    min_swaps = math.inf
    for i in reversed(range(1, N+1)):
        dp[i] = min(dp[i], min_swaps+1) # +1 for breaking
        min_swaps = min(min_swaps, dp[i])
    return ' '.join(map(str, [x -1 for x in dp[1:]]))
    

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')