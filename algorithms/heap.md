# Heap

## min heap implementation 

This is a good heap implementation that allows the ability to delete values from the heap.  

```py
from collections import defaultdict
import heapq

class Heap:
    def __init__(self):
        self.heap = []
        self.deleted = defaultdict(int)
 
    def push(self, val):
        heapq.heappush(self.heap, val)
 
    def clean(self):
        while len(self.heap) > 0 and self.heap[0] in self.deleted:
            self.deleted[self.heap[0]] -= 1
            if self.deleted[self.heap[0]] == 0:
                del self.deleted[self.heap[0]]
            heapq.heappop(self.heap)
 
    def __len__(self):
        self.clean()
        return len(self.heap)
    
    def min(self):
        self.clean()
        return self.heap[0]
    
    def __repr__(self):
        return str(self.deleted)
    
    def delete(self, val):
        self.deleted[val] += 1
 
    def pop(self):
        self.clean()
        return heapq.heappop(self.heap)
```


## finding the next smallest subsequence sum

This algorithm involves a minheap, and can be visualized as a tree, where it follows some important properties.
Where all elements of nums is non-negative integers. 
Finds the kth smallest subsequence sum. 

```cpp
sort(nums.begin(), nums.end());
priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> minheap;
minheap.emplace(nums[0], 0);
long long subt = 0;
while (--k) {
    auto [v, i] = minheap.top();
    minheap.pop();
    subt = v;
    if (i + 1 < N) {
        minheap.emplace(v + nums[i + 1], i + 1);
        minheap.emplace(v + nums[i + 1] - nums[i], i + 1);
    }
}
```