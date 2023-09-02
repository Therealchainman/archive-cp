# Leetcode Weekly Contest 354

## 2778. Sum of Squares of Special Elements

### Solution 1:  sum + enumerate

```py
class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        res = sum(v*v for i, v in enumerate(nums, start = 1) if n % i == 0)
        return res
```

## 2779. Maximum Beauty of an Array After Applying Operation

### Solution 1:  sort + linear scan

```py
class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            s, e = num - k, num + k
            events.append((s, 1))
            events.append((e + 1, -1))
        events.sort()
        res = cnt = 0
        for _, delta in events:
            cnt += delta
            res = max(res, cnt)
        return res
```

### Solution 2:  sliding window + math + optimized sliding window

This is optimized sliding window that is not each window is valid, but it can consider nonvalid windows that are same length as the largest window found so far, and it will find longer windows it increases that size.  

This uses the fact that if nums is sorted and 
nums[l] + k < nums[r] + k
Then for it to be valid it is required that they overlap in this method, nums[r] - k <= nums[l] + k
which gives nums[r] - nums[l] <= 2*k

```py
class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        n = len(nums)
        for right in range(n):
            if nums[right] - nums[left] > 2 * k:
                left += 1
        return right - left + 1
```

## 2780. Minimum Index of a Valid Split

### Solution 1:  prefix and suffix count

```py
class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        freq = Counter(nums)
        dominant = [k for k, v in freq.items() if 2 * v > n][0]
        pcount, scount = 0, freq[dominant]
        for i in range(n):
            pcount += nums[i] == dominant
            scount -= nums[i] == dominant
            if 2 * pcount > i + 1 and 2 * scount > n - i - 1:
                return i
        return -1
```

## 2781. Length of the Longest Valid Substring

### Solution 1:  sliding window + reverse string + set

It's kind of an optimized slidinw window, cause it only needs to check at most 10 characters, so can construct the suffix from that and check existence in forbidden set, The only caveat is need to reverse evertying in forbidden because you are iterating through window in reverse, so the strings are reversed, but you just want to find if a suffix of the current window is in forbidden and then move the left pointer to remove that 

xxxxxxyyy
       ^
where yyy is foribbiden and this is the current window, this is the place to move the pointer and the new window is yy

```py
class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        n = len(word)
        forbidden = set(map(lambda s: s[::-1], forbidden))
        left = res = 0
        window = deque()
        for right in range(n):
            window.append((right, word[right]))
            suffix = ""
            for index, s in reversed(window):
                suffix += s
                if suffix in forbidden:
                    left = index + 1
                    break
            if len(window) == 10:
                window.popleft()
            while window and window[0][0] < left:
                window.popleft()
            res = max(res, right - left + 1)
        return res
```

```py
class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        n = len(word)
        forbidden = set(forbidden)
        left = res = 0
        last = n
        for left in reversed(range(n)):
            for right in range(left, min(n, left + 10, last)):
                if word[left : right + 1] in forbidden: 
                    last = right
                    break
            res = max(res, last - left)
        return res
```

### Solution 2: trie

```py
class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        n = len(word)
        TrieNode = lambda: defaultdict(TrieNode)
        root = TrieNode()
        for w in forbidden:
            reduce(dict.__getitem__, w, root)['word'] = True
        left = res = 0
        last = n
        for left in reversed(range(n)):
            cur = root
            for right in range(left, min(n, left + 10, last)):
                cur = cur[word[right]]
                if cur['word']:
                    last = right
                    break
            res = max(res, last - left)
        return res
```