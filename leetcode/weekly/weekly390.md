# Leetcode Weekly Contest 390

## 3091. Apply Operations to Make Sum of Array Greater Than or Equal to k

### Solution 1:  simulation

```py
class Solution:
    def minOperations(self, k: int) -> int:
        def ceil(x, y):
            return (x + y - 1) // y
        ans = k
        for i in range(1, k + 1):
            ans = min(ans, i + ceil(k, i) - 2)
        return ans
```

## 3092. Most Frequent IDs

### Solution 1:  max heap, counter

```py
class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        n = len(nums)
        maxheap = []
        counts = Counter()
        ans = [0] * n
        for i in range(n):
            counts[nums[i]] += freq[i]
            heappush(maxheap, (-counts[nums[i]], nums[i]))
            while maxheap and counts[maxheap[0][1]] != -maxheap[0][0]: heappop(maxheap)
            ans[i] = -maxheap[0][0]
        return ans
```

## 3093. Longest Common Suffix Queries

### Solution 1:  trie data structure, reverse through the words

```py
class TrieNode(defaultdict):
    def __init__(self):
        super().__init__(TrieNode)
        self.len = math.inf
        self.index = None

    def __repr__(self) -> str:
        return f"TrieNode(len = {self.len}, index = {self.index})"
class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        trie = TrieNode()
        n = len(wordsContainer)
        for i in reversed(range(n)):
            word = wordsContainer[i]
            node = trie
            if len(word) <= node.len:
                node.len = len(word)
                node.index = i
            for ch in reversed(word):
                node = node[ch]
                if len(word) <= node.len:
                    node.len = len(word)
                    node.index = i
        m = len(wordsQuery)
        ans = [0] * m
        for i, word in enumerate(wordsQuery):
            node = trie
            ans[i] = node.index
            for ch in reversed(word):
                node = node[ch]
                if node.len < math.inf: ans[i] = node.index
        return ans
```

### Solution 2:  double hashing, rolling hash, rabin karp 

```py
class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        p, MOD1, MOD2 = 31, int(1e9) + 7, int(1e9) + 9
        coefficient = lambda x: ord(x) - ord('a') + 1
        shashes = {}
        add = lambda h, mod, ch: ((h * p) % mod + coefficient(ch)) % mod
        n = len(wordsContainer)
        for i in reversed(range(n)):
            word = wordsContainer[i]
            hash1 = hash2 = 0
            if len(word) <= shashes.get((hash1, hash2), (0, math.inf))[1]:
                shashes[(hash1, hash2)] = (i, len(word))
            for ch in reversed(word):
                hash1 = add(hash1, MOD1, ch)
                hash2 = add(hash2, MOD2, ch)
                if len(word) <= shashes.get((hash1, hash2), (0, math.inf))[1]:
                    shashes[(hash1, hash2)] = (i, len(word))
        m = len(wordsQuery)
        ans = [0] * m
        for i, word in enumerate(wordsQuery):
            hash1 = hash2 = 0
            ans[i] = shashes[(hash1, hash2)][0]
            for ch in reversed(word):
                hash1 = add(hash1, MOD1, ch)
                hash2 = add(hash2, MOD2, ch)
                if (hash1, hash2) in shashes: ans[i] = shashes[(hash1, hash2)][0]
        return ans
```

