# Leetcode Weekly Contest 385

## 3043. Find the Length of the Longest Common Prefix

### Solution 1:  set

Because the there are at most 8 characters in the sequences, you can really brute force this one, store all the prefixes in a set and just check which ones exist in the set.  Cause at most there will be 8 * 50,000 = 400,000 distinct prefixes.

```py
class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        seen = set()
        for num in arr2:
            while num > 0:
                seen.add(num)
                num //= 10
        ans = 0
        for num in arr1:
            while num > 0:
                if num in seen: 
                    ans = max(ans, len(str(num)))
                    break
                num //= 10
        return ans
```

## 3044. Most Frequent Prime

### Solution 1:  number theory, prime detection, simulation

Since the grid is so small 6 by 6.  The integer must be less than 10^6.  There will be at most about 36 * 6 possible integers formed. Which is 216.  So actually it makes more sense to use the sqrt(n) technique to detect prime, instead of precomputing with prime sieve.  Cause the sqrt(n) at worse will be 10^3.  so that is plenty fast enough.  Actually probably faster than precomputing.

```py
def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
class Solution:
    def mostFrequentPrime(self, mat: List[List[int]]) -> int:
        R, C = len(mat), len(mat[0])
        counts = Counter()
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        def calc(r, c, d):
            num = length = 0
            while in_bounds(r, c):
                num = num * 10 + mat[r][c]
                length += 1
                if length > 1 and is_prime(num): counts[num] += 1
                r += DIRECTIONS[d][0]
                c += DIRECTIONS[d][1]
        for r, c in product(range(R), range(C)):
            for d in range(8):
                calc(r, c, d)
        max_f = max(counts.values(), default = 0)
        ans = -1
        for p, c in counts.items():
            if c == max_f: ans = max(ans, p)
        return ans
```

## 3045. Count Prefix and Suffix Pairs II

### Solution 1:  Trie, prefix and suffix matching

Since the trie will not be too large cause the sum of characters is at most 500,000 it should be fast enough. 

```py
class TrieNode(defaultdict):
    def __init__(self):
        super().__init__(TrieNode)
        self.count = 0 # how many words have this pair
    def __repr__(self) -> str:
        return f'is_word: {self.is_word} prefix_count: {self.prefix_count}, children: {self.keys()}'
class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        trie = TrieNode()
        ans = 0
        for w in reversed(words):
            # search trie
            node = trie
            for c1, c2 in zip(w, reversed(w)):
                node = node[(c1, c2)]
            ans += node.count
            # add to trie
            node = trie
            for c1, c2 in zip(w, reversed(w)):
                node = node[(c1, c2)]
                node.count += 1
        return ans
```