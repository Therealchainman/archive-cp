# Leetcode Weekly Contest 112

## 2839. Check if Strings Can be Made Equal With Operations I

### Solution 1:  brute force + bitmask

bitmask is used to represent 4 possible configurations
0, 1, 2, 3.  either you swap none, swap at index 0, swap at index 1, or swap at both index 0 and index 1

```py
class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        for mask1 in range(1 << 2):
            for mask2 in range(1 << 2):
                tmp1, tmp2 = list(s1), list(s2)
                for i in range(2):
                    if (mask1 >> i) & 1:
                        tmp1[i], tmp1[i + 2] = tmp1[i + 2], tmp1[i]
                    if (mask2 >> i) & 1:
                        tmp2[i], tmp2[i + 2] = tmp2[i + 2], tmp2[i]
                if tmp1 == tmp2: return True
        return False
```

## 2840. Check if Strings Can be Made Equal With Operations II

### Solution 1:  greedy + sort

The trick here is realizing tha you can make any rearrangement of characters under modulus 2 of the index.  
That is 0 1 2 3 4 5 
You can realize you can swap any character between the 0,2,4 cause they will all have even difference. 
and anything at 1, 3, 5 will also have even difference, so you can split the characters up into even and odds and just sort them. 
If both can have the same string after sorting in this way then there are some number of operations that can make them equal. 

```py
class Solution:
    def checkStrings(self, s1: str, s2: str) -> bool:
        def rearrange(s):
            n = len(s)
            odd, even = [], []
            for i in range(n):
                if i & 1:
                    odd.append(s[i])
                else:
                    even.append(s[i])
            odd.sort(reverse = True)
            even.sort(reverse = True)
            res = []
            while odd or even:
                if even:
                    res.append(even.pop())
                if odd:
                    res.append(odd.pop())
            return "".join(res)
        s1 = rearrange(s1)
        s2 = rearrange(s2)
        return s1 == s2
```

## 2841. Maximum Sum of Almost Unique Subarray

### Solution 1:  fixed sliding window + sum + distinct count

```py
class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        res = wcount = wsum = 0
        freq = Counter()
        for i in range(n):
            freq[nums[i]] += 1
            wsum += nums[i]
            if freq[nums[i]] == 1: wcount += 1
            if i >= k - 1:
                if wcount >= m: res = max(res, wsum)
                left_elem = nums[i - k + 1]
                freq[left_elem] -= 1
                wsum -= left_elem
                if freq[left_elem] == 0: wcount -= 1
        return res
```

## 2842. Count K-Subsequences of a String With Maximum Beauty

### Solution 1: math + combinations + greedy + sort

To maximize the beauty it is always best to take the highest frequency characters.  There are going to be at most 26 characters, so you can know when to return 0 based on the inputs. 

Sort the frequency of the characters, and divide the frequency into a prefix and a suffix, where the prefix are all the frequencies that are larger than the kth largest frequency in the array, and the suffix is all frequencies equal to the kth largest frequency, up to it

for example given a frequency values
k = 5
3 2 2 1 1 1 1 0 0 0 
p p p s s t t
p means it is part of prefix, for the prefix you just multiple each one and calculate the prefix_prod or pprod, since there are three characters to choose multiplied by 2 characters to choose multiplied by 2 characters to choose.
s means it is part of the suffix, these are all equal to the kth largest frequency and are within the first k
t means it is part of the options you can choose for the suffix, you can actually choose any 2 out of the 4 ones in this example.   That is just a combination problem.  It is how many combinations are there picking s elements from s + t elements. So you can use the math.comb function to calculate that.  What you want to do with that is that for every combination you will have ssprod ways to choose those combinations that is for the ones here you have 1 * 1 + 1 * 1 + 1 * 1 + 1 * 1 + 1 * 1 + 1 * 1= 1 * (1 + 1 + 1 + 1 + 1 + 1), cause there are three combinations for pick 2 from 4 elements.  Since there are only 1 element to pick, but suppose instead of ones it was twos.  Then you know there are 2 * 2 possible ways to pick elements from the last two elements.  So in that case it would be 2 * 2 * (6).  


```py
class Solution:
    def countKSubsequencesWithMaxBeauty(self, s: str, k: int) -> int:
        if k > len(set(s)) : return 0
        mod = int(1e9) + 7
        freq = [0] * 26
        unicode = lambda ch: ord(ch) - ord('a')
        for ch in s:
            freq[unicode(ch)] += 1
        freq.sort(reverse = True)
        scount = 0
        pprod = sprod = 1
        for i in range(k):
            if freq[i] == freq[k - 1]: 
                scount += 1
                pprod = (pprod * freq[i]) % mod
            elif freq[i] > freq[k - 1]:
                sprod = (sprod * freq[i]) % mod
        tcount = freq.count(freq[k - 1])
        res = (sprod * pprod * math.comb(tcount, scount)) % mod
        return res
```

### Solution 2: math + combinations + power

This is actually using same idea as above but is a more general math equation that works for it

current = current * cnt^take * combinations(ffreq[cnt], take)

cause 1 * 1 * 1 = 1^3, and if you can do it however many ways you want to multiply by that. 

And if it is 2 * 2 * 2 = 2^3 so you pick from those and then you multiply that by number of combinations you can pick with twos. 

```py
class Solution:
    def countKSubsequencesWithMaxBeauty(self, s: str, k: int) -> int:
        mod = int(1e9) + 7
        freq = Counter(s)
        n = len(s)
        if k > len(freq): return 0
        ffreq = [0] * (n + 1)
        for f in freq.values():
            ffreq[f] += 1
        res = 1
        for cnt in reversed(range(n + 1)):
            if ffreq[cnt] == 0: continue
            take = min(k, ffreq[cnt])
            res = (res * pow(cnt, take, mod) * math.comb(ffreq[cnt], take)) % mod
            k -= take
            if k == 0: break
        return res
```

