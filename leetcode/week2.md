## 271. Encode and Decode Strings

### Solution 1:  Chunked Transfer Encoding + string

Stores the size of each chunk as a prefix and then there is the delimiter '#'.

```py
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        n = len(strs)
        result = [None] * n
        for i in range(n):
            size = len(strs[i])
            result[i] = f"{size}#{strs[i]}"
        return "".join(result)

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        result = []
        i = 0
        while i < len(s):
            j = s.find('#', i)
            size = int(s[i:j])
            str_ = s[j + 1:j + 1 + size]
            result.append(str_)
            i = j + 1 + size
        return result
        
# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))
```

## 920. Number of Music Playlists

### Solution 1:  dynamic programming

dp[i][j] = number of playlists of length i that have exactly j unique songs

Two transition states:
1. play song
Play a new song that has not been played before this will increase the distinct songs by one
You need to determine how many songs you can play in this instance, it is going to be the total number of songs - number of unique songs played so far + 1. For example, 
xxxx 4 unique songs played, and there are n = 10 total songs, then you can play either song 5,6,7,8,9,10, which is 10 - 4 = 6 songs
The transition state looks like dp[i - 1][j - 1] * (n - (j - 1)), because you are coming from state with j - 1 unique songs played.
2. replay song
Play a song that has been played before, this will not increase the distinct songs played
Because you can only play a replayed song after k other songs you need to consider this
if j = 5, so 5 unique songs played 
and k = 3, so you need to play 3 songs between so for instance, 1,2,3,4,5,x => you can only replay 1 and 2 so that means there are 2 songs you can replay for 5 unique songs always, 
because there must be 3 songs in window that are distinct, so j - k is the songs you can play in this scenario so multipy by that
you get dp[i - 1][j] * (j - k)

why is it multiplication? 
Because at each state you have x possible songs you can play, and so if there are 4 ways to get to thst state, you can now take those 4 ways call them x1, x2, x3, x4
and if x = 3
you can now add 1 to end of all 4 states and 2 to end of all 4 states and 3 to end of all 4 states, so that is 3 * 4 = 12, or x * num_ways
another way I think of it is take this

_ _ _ _ _
        ^
      4
so you know there are 4 ways to fill in the first 4 slots, now for the current slot you are at, if you have x choices, then you are going to add it to end of all the previous 4 ways, so you get 4 * x ways now


```py
class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        mod = int(1e9) + 7
        dp = [[0] * (n + 1) for _ in range(goal + 1)]
        dp[0][0] = 1
        for i, j in product(range(1, goal + 1), range(1, n + 1)):
            dp[i][j] = (dp[i - 1][j - 1] * (n - j + 1) + dp[i - 1][j] * max(j - k, 0)) % mod
        return dp[goal][n]
```

### Solution 2:  math + combinatorics + inclusion exclusion principle

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```