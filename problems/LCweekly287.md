# Leetcode Weekly contest 287

## Summary

I did this contest in virtual mode.  I was able to solve the first 3 questions slow and steady. 
The last question I coded a long trie solution from scratch.  Which end up just being TLE.  I did
not find the preprocess till at end of contest. 

## 2224. Minimum Number of Operations to Convert Time

### Solution 1: Greedy

convert to minutes and use the larger time increments first.

```py
class Solution:
    def convertTime(self, current: str, correct: str) -> int:
        get_minutes = lambda arr: 60*int(arr[0])+int(arr[1])
        current_minutes = get_minutes(current.split(':'))
        correct_minutes = get_minutes(correct.split(':'))
        delta = correct_minutes - current_minutes
        cnt = 0
        for d in [60,15,5,1]:
            num_times = delta // d
            cnt += num_times
            delta -= num_times*d
        return cnt
```

## 2225. Find Players With Zero or One Losses

### Solution 1: Use a table to store the count of losses for each player

We want to return players with 0 and 1 loss

```py
class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        winners, losers, table = [],[],{}
        for winner, loser in matches:
            table[winner] = table.get(winner,0)
            table[loser] = table.get(loser,0) + 1
        for k, v in table.items():
            if v==0:
                winners.append(k)
            elif v==1:
                losers.append(k)
        return [sorted(winners),sorted(losers)]
```

### Solution 2: Same idea using a counter for losses and set for winners + set difference to get 0 losses

```py
class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        winners, losers= set(), Counter()
        for winner, loser in matches:
            winners.add(winner)
            losers[loser] += 1
        one_losses = [player for player, losts in losers.items() if losts==1]
        return [sorted(list(winners-set(losers.keys()))), sorted(one_losses)]
```

## 2225. Find Players With Zero or One Losses
 
### Solution 1: binary search for the maximum number of candies to give to k children

```py
class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        left, right = 0, sum(candies)//k
        while left < right:
            mid = (left+right+1)>>1
            if sum(candy//mid for candy in candies) >=k:
                left = mid
            else:
                right = mid - 1
        return left
```

## 2227. Encrypt and Decrypt Strings

### Solution 1: hash map for encrypt and counter for decrypt + preprocess decryption of dictionary

```py
class Encrypter:
    def __init__(self, keys: List[str], values: List[str], dictionary: List[str]):
        self.encrypt_map = {k: v for k,v in zip(keys,values)}
        self.decrypt_map = Counter()
        for word in dictionary:
            self.decrypt_map[self.encrypt(word)]+=1

    def encrypt(self, word1: str) -> str:
        return "".join(map(lambda x: self.encrypt_map[x], word1))

    def decrypt(self, word2: str) -> int:
        return self.decrypt_map[word2]
```

### Solution 2: hash map for encrypt and decrypt + trie data structure for dictionary

Trie should work but it is TLE right now, needs to be optimized

TODO: Optimize this trie, prune search etc.

```py
class Node:
    def __init__(self):
        self.children = [0]*26
        self.is_leaf = False 
class Trie:
    def __init__(self):
        self.trie = [Node()]
    def add(self, s):
        cur = 0
        for ch in s:
            i = ord(ch)-ord('a')
            if self.trie[cur].children[i]==0:
                self.trie[cur].children[i] = len(self.trie)
                self.trie.append(Node())
            cur = self.trie[cur].children[i]
        self.trie[cur].is_leaf = True
    def match_count(self, s, decrypt_map):
        cnt = 0
        dq = deque()
        for v in decrypt_map[s[0]]:
            if self.trie[0].children[ord(v)-ord('a')]:
                dq.append((v,0,1))
        while dq:
            ch, cur, index = dq.popleft()
            ch_val = ord(ch)-ord('a')
            if index == len(s):
                cur = self.trie[cur].children[ch_val]
                cnt += 1 if self.trie[cur].is_leaf else 0
                continue
            for v in decrypt_map[s[index]]:
                ncur = self.trie[cur].children[ch_val]
                if self.trie[ncur].children[ord(v)-ord('a')]:
                    dq.append((v, ncur, index+1))
        return cnt
class Encrypter:
    def __init__(self, keys: List[str], values: List[str], dictionary: List[str]):
        self.encrypt_map = {k: v for k,v in zip(keys,values)}
        self.decrypt_map = defaultdict(list)
        for k, v in zip(keys,values):
            self.decrypt_map[v].append(k)
        self.prefix_tree = Trie()
        for word in dictionary:
            self.prefix_tree.add(word)

    def encrypt(self, word1: str) -> str:
        return "".join(map(lambda x: self.encrypt_map[x], word1))

    def decrypt(self, word2: str) -> int:
        word = [word2[index:index+2] for index in range(0,len(word2),2)]
        return self.prefix_tree.match_count(word, self.decrypt_map)


# Your Encrypter object will be instantiated and called as such:
# obj = Encrypter(keys, values, dictionary)
# param_1 = obj.encrypt(word1)
# param_2 = obj.decrypt(word2)
```