# Leetcode Biweekly Contest 68

## 2114. Maximum Number of Words Found in Sentences

### Solution: get the maximum result by splitting all strings with the ' ' whitespace character into lists and get the length

```py
def mostWordsFound(self, sentences: List[str]) -> int:
    return max(len(sentence.split(' ')) for sentence in sentences)
```

```c++
int mostWordsFound(vector<string>& sentences) {
    int mx = 0;
    for (string& s : sentences) {
        stringstream ss(s);
        string tmp;
        int cnt = 0;
        while (getline(ss,tmp,' ')) {
            cnt++;
        }
        mx = max(mx,cnt);
    }
    return mx;
}
```

## 2115. Find All Possible Recipes from Given Supplies

### Solution: topological sort and bfs through the directed graph

```c++
vector<string> findAllRecipes(vector<string>& recipes, vector<vector<string>>& ingredients, vector<string>& supplies) {
    vector<string> results;
    int n = recipes.size();
    unordered_map<string, int> indegrees;
    unordered_map<string, vector<string>> graph;
    for (int i = 0;i<n;i++) {
        for (string& ingred : ingredients[i]) {
            graph[ingred].push_back(recipes[i]);
            indegrees[recipes[i]]++;
        }
    }
    queue<string> q;
    for (string &sup : supplies) {
        q.push(sup);
    }
    while (!q.empty()) {
        string ingredient = q.front();
        q.pop();
        for (auto& nei : graph[ingredient]) {
            if (--indegrees[nei]==0) {
                q.push(nei);
                results.push_back(nei);
            }
        }
    }
    return results;
}
```

## 2116. Check if parentheses String Can Be Valid


### Solution: 

```c++

```

"((()(()()))()((()()))))()((()(()"
"10111100100101001110100010001001"


## 2117. Abbreviating the Product of a Range

### Solution 1: Count the number of trailing zeroes for factorial(right)-factorial(left-1) and compute the prefix and suffix. Brute force if 
fewer than or equal to 10 digits.  

We don't want to count trailing zeroes in the fewer than 10 digits. 

The tricky part for me was computing the prefix.  I kind of just used a decimal or double in c++.  To compute the prefix
where I kept at least 5 digits before the decimal point.  Then compute it like that.  But to be frank it is a bit ignoreing precision and 
other errors potentially.  

```c++
const int MOD = 1e5;
class Solution {
public:
    int trailingZeroes(int n) {
        int cntFives = 0;
        for (int i = 5;i<=n;i*=5) {
            cntFives += (n/i);
        }
        return cntFives;
    }
    string abbreviateProduct(int left, int right) {
        long long prod = 1;
        int es = 0;
        for (long long i = left;i<=right;i++) {
            prod*=i;
            while (prod%10==0 && prod>0) {
                prod/=10;
                es++;
            }
            if (to_string(prod).size()>10) {
                break;
            }
        }
        if (to_string(prod).size()<=10) {
            return to_string(prod) + 'e' + to_string(es);
        }
        // solve the difficult case that has d>10, so it has prefix and suffix.  
        int zeroes = trailingZeroes(right)-trailingZeroes(left-1);
        int cntFives = zeroes, cntTwos = zeroes;
    
        // compute the suffix
        prod = 1;
        for (long long i = left;i<=right;i++) {
            prod*=i;
            while (prod%5==0 && cntFives>0) {
                prod/=5;
                cntFives--;
            }
            while (prod%2==0 && cntTwos>0) {
                prod/=2;
                cntTwos--;
            }
            prod%=MOD;
        }
        int leadingZeroes = 5-to_string(prod).size();

        string suffix = "";
        while (leadingZeroes--) {
            suffix += '0';
        }
        suffix += to_string(prod);
        // compute the prefix
        double pprod = 1.0;
        for (long long i = left;i<=right;i++) {
            pprod*=i;
            while (pprod>MOD) {
                pprod/=10.0;
            }
        }
        string prefix = to_string(int(pprod));
        return prefix + "..." + suffix + 'e'+to_string(zeroes);
    }
};
```

# Leetcode Biweekly Contest 69

# Leetcode Biweekly Contest 70

## 2144. Minimum Cost of Buying Candies With Discount

### Solution: modular math + array iteration + sort

```c++
int minimumCost(vector<int>& cost) {
    sort(cost.begin(),cost.end());
    int n = cost.size(), sumCost = 0;
    for (int i = 0;i<n;i++) {
        sumCost += (i%3==n%3 ? 0 : cost[i]);
    }
    return sumCost;
}
```

## 2145. Count the Hidden Sequences

### Solution: math bounds

```c++
int numberOfArrays(vector<int>& D, int lower, int upper) {
    long long mn = 0, mx = 0, num = 0;
    for (int i=0;i<D.size();i++) {
        num+=D[i];
        mn = min(mn,num);
        mx = max(mx,num);
    }
    
    mx += (lower-mn);
    return max(0LL,(long long)upper-mx+1LL);
}
```

## 2146. K Highest Ranked Items Within a Price Range

### Solution: BFS + custom sort

TC: O(mnlog(mn))

```c++
struct Item {
    int row, col, price, dist;
    void init(int r, int c, int p, int d) {
        row = r, col = c, price = p, dist = d;
    }
};
class Solution {
public:
    vector<vector<int>> highestRankedKItems(vector<vector<int>>& grid, vector<int>& pricing, vector<int>& start, int k) {
        vector<Item> items;
        int R = grid.size(), C = grid[0].size(), low = pricing[0], high = pricing[1];
        queue<Item> q;
        Item sitem;
        sitem.init(start[0],start[1],grid[start[0]][start[1]],0);
        auto inPrice = [&](const int& i, const int& j) {
            return grid[i][j]>=low && grid[i][j]<=high;
        };
        if (inPrice(start[0],start[1])) {
            items.push_back(sitem);
        }
        grid[start[0]][start[1]]=-1;
        q.push(sitem);
        auto inBounds = [&](const int& i, const int& j) {
            return i>=0 && i<R && j>=0 &&j<C;
        };
        while (!q.empty()) {
            Item curItem = q.front();
            q.pop();
            for (int dr = -1;dr<=1;dr++) {
                for (int dc =-1;dc<=1;dc++) {
                    if (abs(dc+dr)!=1) continue;
                    int nr = curItem.row+dr, nc = curItem.col+dc;
                    if (!inBounds(nr,nc) || grid[nr][nc]==-1) continue;
                    if (grid[nr][nc]>0) {
                        Item item;
                        item.init(nr,nc,grid[nr][nc],curItem.dist+1);
                        q.push(item);
                        if (inPrice(nr,nc)) {
                            items.push_back(item);
                        }
                    }
                    grid[nr][nc]=-1; // set as visited
                }
            }
        }
        sort(items.begin(),items.end(),[&](const Item& a, const Item& b) {
            if (a.dist != b.dist) return a.dist < b.dist;
            if (a.price != b.price) return a.price < b.price;
            if (a.row != b.row) return a.row < b.row;
            return a.col < b.col;
        });
        vector<vector<int>> res;
        for (int i = 0;i<k && i<items.size();i++) {
            res.push_back({items[i].row,items[i].col});
        }
        return res;
    }
```

### Solution: BFS + priority queue + custom sort

TC: O(mnlog(k))

```c++
struct Item {
    int row, col, price, dist;
    void init(int r, int c, int p, int d) {
        row = r, col = c, price = p, dist = d;
    }
};
struct compare {
    bool operator()(const Item& a, const Item& b) {
        if (a.dist != b.dist) return a.dist < b.dist;
        if (a.price != b.price) return a.price < b.price;
        if (a.row != b.row) return a.row < b.row;
        return a.col < b.col;
    }  
};
class Solution {
public:
    vector<vector<int>> highestRankedKItems(vector<vector<int>>& grid, vector<int>& pricing, vector<int>& start, int k) {
        priority_queue<Item,vector<Item>,compare> heap;
        int R = grid.size(), C = grid[0].size(), low = pricing[0], high = pricing[1];
        queue<Item> q;
        Item sitem;
        sitem.init(start[0],start[1],grid[start[0]][start[1]],0);
        auto inPrice = [&](const int& i, const int& j) {
            return grid[i][j]>=low && grid[i][j]<=high;
        };
        if (inPrice(start[0],start[1])) {
            heap.push(sitem);
        }
        grid[start[0]][start[1]]=-1;
        q.push(sitem);
        auto inBounds = [&](const int& i, const int& j) {
            return i>=0 && i<R && j>=0 &&j<C;
        };
        while (!q.empty()) {
            Item curItem = q.front();
            q.pop();
            for (int dr = -1;dr<=1;dr++) {
                for (int dc =-1;dc<=1;dc++) {
                    if (abs(dc+dr)!=1) continue;
                    int nr = curItem.row+dr, nc = curItem.col+dc;
                    if (!inBounds(nr,nc) || grid[nr][nc]==-1) continue;
                    if (grid[nr][nc]>0) {
                        Item item;
                        item.init(nr,nc,grid[nr][nc],curItem.dist+1);
                        q.push(item);
                        if (inPrice(nr,nc)) {
                            heap.push(item);
                        }
                        if (heap.size()>k) {
                            heap.pop();
                        }
                    }
                    grid[nr][nc]=-1; // set as visited
                }
            }
        }
        vector<vector<int>> res;
        while (!heap.empty()) {
            Item item = heap.top();
            heap.pop();
            res.push_back({item.row,item.col});
        }
        reverse(res.begin(),res.end());
        return res;
    }
};
```

## 2147. Number of Ways to Divide a Long Corridor

### Solution: greedy solution with combinatorics 

```c++
const int MOD = 1e9+7;
const char SEAT = 'S';
class Solution {
public:
    int numberOfWays(string corridor) {
        int cntSeats = count_if(corridor.begin(),corridor.end(),[&](const auto& a) {return a==SEAT;}), n = corridor.size();
        if (cntSeats%2!=0 || cntSeats==0) {return 0;}
        long long way = 0, totalWays = 1;
        for (int i = 0, seats = 0;i<n;i++) {
            seats += (corridor[i]==SEAT);
            if (seats==3) {
                totalWays = (totalWays*way)%MOD;
                seats = 1;
                way = 0;
            }
            way += (seats==2);
        }
        return totalWays;
    }
};
```

# Leetcode Biweekly Contest 71




## 2165. Smallest Value of the Rearranged Number

### Solution:


```py
class Solution:
    def smallestNumber(self, num: int) -> int:
        if num==0: return num
        sign = False if num<0 else True
        smallest = math.inf
        if sign:
            num = list(str(num))
            num.sort()
            cntZeros = sum(1 for x in num if x=='0')
            return int("".join(num[cntZeros:cntZeros+1] + num[:cntZeros]+num[cntZeros+1:]))
        num = list(str(num)[1:])
        num.sort(reverse=True)
        return -int("".join(num))
```

## 2166. Bitset Design

### Solution: datastructure impelementation with lists of 0 and 1s, storing bits and flipped bits, and needed variables

```py
class Bitset:

    def __init__(self, size: int):
        self.bits = [0 for _ in range(size)]
        self.fbits = [1 for _ in range(size)]
        self.cnt_ones = 0
        self.cnt_ones_flipped = size
        self.size = size

    def fix(self, idx: int) -> None:
        if self.bits[idx]==0:
            self.bits[idx]=1
            self.fbits[idx]=0
            self.cnt_ones += 1
            self.cnt_ones_flipped-=1

    def unfix(self, idx: int) -> None:
        if self.bits[idx]==1:
            self.bits[idx]=0
            self.fbits[idx]=1
            self.cnt_ones-=1
            self.cnt_ones_flipped+=1

    def flip(self) -> None:
        self.bits, self.fbits = self.fbits, self.bits
        self.cnt_ones, self.cnt_ones_flipped = self.cnt_ones_flipped, self.cnt_ones

    def all(self) -> bool:
        return self.cnt_ones==self.size

    def one(self) -> bool:
        return self.cnt_ones>0

    def count(self) -> int:
        return self.cnt_ones

    def toString(self) -> str:
        return "".join(map(str,self.bits))
```

# Leetcode Biweekly Contest 72

## 2176. Count Equal and Divisible Pairs in an Array

### Solution: Brute force two for loops

```py
class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        cnt = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                cnt += 1 if nums[i]==nums[j] and (i*j)%k==0 else 0
        return cnt
```

## 2177. Find Three Consecutive Integers That Sum to a Given Number

### Solution: math, if divisible by 3 take the surrounding two elements

```py
class Solution:
    def sumOfThree(self, num: int) -> List[int]:
        if num%3!=0: return []
        return [num//3-1, num//3, num//3+1]
```


## 2178. Maximum Split of Positive Even Integers

### Solution: Greedy, find all elements that sum above and remove the single element if possible

```py
class Solution:
    def maximumEvenSplit(self, finalSum: int) -> List[int]:
        """
        if you start from 2, and include all positive unique even integers, at most there could be 100,000.
        This is a good threshold to know cause that means my result can only be that long. 
        """
        ans = []
        sum_ = 0
        for i in range(2, finalSum+1, 2):
            ans.append(i)
            sum_ += i
            if sum_ >= finalSum: break
        if sum_ == finalSum: return ans
        if (sum_ - finalSum)%2==0:
            ans.remove(sum_ - finalSum)
            return ans
        return []
```

## 2179. Count Good Triplets in an Array

### Solution: IDK

```py

```

# Leetcode Biweekly Contest 73


## 2190. Most Frequent Number Following Key In an Array

### Solution: Counter + Hashmap

```py
class Solution:
    def mostFrequent(self, nums: List[int], key: int) -> int:
        c = Counter()
        for i in range(1, len(nums)):
            if nums[i-1]==key:
                c[nums[i]] += 1
        most = c.most_common()
        return most[0][0]
```

## 2191. Sort the Jumbled Numbers

### Solution: custom sort

```py
class Solution:
    def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
        def map_sort(num):
            now = 0
            for x in str(num):
                x = ord(x)-ord('0')
                now = now*10+mapping[x]
            return now
                
        nums.sort(key=map_sort)
        return nums
```


```py
class Solution:
    def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
        def map_sort(num):
            now = 0
            for x in str(num):
                x = ord(x)-ord('0')
                now = now*10+mapping[x]
            return now
        nums = zip(nums, [map_sort(num) for num in nums])
        return [x for x,y in sorted(nums, key=lambda x: x[1])]
```

## 2192. All Ancestors of a Node in a Directed Acyclic Graph

### Solution: DFS + iterate through each ancestor in order and add it to answer for each node you can reach in dfs

```py
class Solution:
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        answer = [[] for _ in range(n)]
        graph = defaultdict(list)
        for fro, to in edges:
            graph[fro].append(to)
        def dfs(ancestor, node):
            for nei in graph[node]:
                if answer[nei] and answer[nei][-1] == ancestor: continue
                answer[nei].append(ancestor)
                dfs(ancestor, nei)
            
        for i in range(n): dfs(i,i)
        return answer
```

## 2193. Minimum Number of Moves to Make Palindrome

### Solution: 2 pointer algorithm + greedily choose the best swap with left or right as the pivot character

"scpcyxprxxsjyjrww"


```py
class Solution:
    def minMovesToMakePalindrome(self, s: str) -> int:
        s = list(s)
        cnt = 0
        n = len(s)
        for L in range(n//2):
            R = n-L-1
            if s[L]!=s[R]:
                right = R
                while s[L] != s[right]:
                    right-=1
                left = L
                while s[left] != s[R]:
                    left += 1
                if R-right < left-L: # it is better to swap from element at right position to the end R position
                    cnt += R-right
                    for k in range(right,R):
                        s[k] = s[k+1]
                else:
                    cnt += left-L
                    for k in range(left, L, -1):
                        s[k] = s[k-1]
                    
        return cnt
```

# Leetcode Biweekly Contest 74

## 2206. Divide Array Into Equal Pairs

### Solution: 

```py
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        counts = Counter(nums)
        for cnt in counts.values():
            if cnt%2==1: return False
        return True
```

```py
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        counts = Counter(nums)
        return not list(filter(lambda cnt: cnt%2==1, counts.values()))
```

## 2207. Maximize Number of Subsequences in a String

### Solution: Greedy 

```py
class Solution:
    def maximumSubsequenceCount(self, text: str, pattern: str) -> int:
        s = "".join([ch for ch in text if ch in pattern])
        def get_max(ss):
            sum_ = cnt = 0
            for ch in ss:
                if ch==pattern[1]:
                    sum_ += cnt
                if ch==pattern[0]:
                    cnt += 1
            return sum_
        return max(get_max(pattern[0] + s), get_max(s + pattern[1]))
```

## 2208. Minimum Operations to Halve Array Sum

### Solution: max heap datastructure

```py
class Solution:
    def halveArray(self, nums: List[int]) -> int:
        nums = [-x for x in nums]
        heapify(nums)
        half = -sum(nums)/2
        cnt = 0
        while half > 0:
            val = -heappop(nums)
            val/=2
            half -= val
            heappush(nums, -val)
            cnt += 1
        return cnt
```

## 2209. Minimum White Tiles After Covering With Carpets

### Solution: recursive DP

```py
class Solution:
    def minimumWhiteTiles(self, floor: str, numCarpets: int, carpetLen: int) -> int:
        n = len(floor)
        suffix = [0]*(n+1)
        for i in range(n-1,-1,-1):
            suffix[i] = suffix[i+1] + (floor[i]=='1')
        @cache
        def dfs(pos, count):
            if pos >= n: return 0
            if count == 0: return suffix[pos]
            return min(dfs(pos+carpetLen, count-1), dfs(pos+1, count) + (floor[pos] == '1'))
        return dfs(0, numCarpets)
```


# Leetcode Biweekly Contest 75

## Summary

## 2220. Minimum Bit Flips to Convert Number

### Solution 1: xor + bit_count

```py
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        return (start ^ goal).bit_count()
```

## 2221. Find Triangular Sum of an Array

### Solution 1: optimized space solution, reuse nums array

```py
class Solution:
    def triangularSum(self, nums: List[int]) -> int:
        for end in range(len(nums)+1)[::-1]:
            for i in range(1, end):
                nums[i-1] = (nums[i-1]+nums[i])%10
        return nums[0]
```

### Solution 2: pairwise to get the two adjacent elements in pair

```py
class Solution:
    def triangularSum(self, nums: List[int]) -> int:
        while len(nums) > 1:
            nums = [(a+b)%10 for a,b in pairwise(nums)]
        return nums[0]
```

## 2222. Number of Ways to Select Buildings

### Solution 1: prefix sums for only 2 patterns 101, 010

```py
class Solution:
    def numberOfWays(self, s: str) -> int:
        n = len(s)
        prefix_zeros, prefix_ones = [0]*(n+1), [0]*(n+1)
        for i in range(n):
            prefix_zeros[i+1] = prefix_zeros[i] + (s[i]=='0')
            prefix_ones[i+1] = prefix_ones[i] + (s[i]=='1')
        ways = 0
        for i in range(n):
            if s[i] == '0':
                ways += prefix_ones[i]*(prefix_ones[-1]-prefix_ones[i])
            if s[i] == '1':
                ways += prefix_zeros[i]*(prefix_zeros[-1]-prefix_zeros[i])
        return ways
```

## 2223. Sum of Scores of Built Strings

### Solution 1: Brute force z-algorithm (z-array)

This will TLE cause it is O(n^2)

```py
class Solution:
    def sumScores(self, s: str) -> int:
        n = len(s)
        z = [0]*n
        for i in range(1, n):
            while z[i]+i < n and s[z[i]+i] == s[z[i]]:
                z[i] += 1
        return sum(z) + n
```

### Solution 2: optimized z-algorithm

```py
class Solution:
    def sumScores(self, s: str) -> int:
        n = len(s)
        z = [0]*n
        z[0] = n
        left = right = 0
        for i in range(1,n):
            if i > right:
                left = right = i
                while right < n and s[right-left] == s[right]:
                    right += 1
                z[i] = right - left
                right -= 1
            else:
                k = i - left
                # IF PREVIOUS MATCHED SEGMENT IS NOT TOUCHING BOUNDARIES OF CURRENT MATCHED SEGMENT
                if z[k] < right - i + 1:
                    z[i] = z[k]
                else:
                    left = i
                    while right < n and s[right-left] == s[right]:
                        right += 1
                    z[i] = right - left
                    right -= 1
        return sum(z)
```


# Leetcode Biweekly Contest 76

## Summary

## 2239. Find Closest Number to Zero

### Solution 1: 

```py
class Solution:
    def findClosestNumber(self, nums: List[int]) -> int:
        return min(nums, key=lambda x: (abs(x),-x))
```

## 2240. Number of Ways to Buy Pens and Pencils

### Solution 1: number of ways with single loop

```py
class Solution:
    def waysToBuyPensPencils(self, total: int, cost1: int, cost2: int) -> int:
        cnt = 0
        while total >= 0:
            cnt += total//cost2 + 1
            total -= cost1
        return cnt
```

## 2241. Design an ATM Machine

### Solution 1: hash table

```py
class ATM:

    def __init__(self):
        self.cash = [0]*5
        self.values = [20,50,100,200,500]        

    def deposit(self, banknotesCount: List[int]) -> None:
        for i, cnt in enumerate(banknotesCount):
            self.cash[i] += cnt

    def withdraw(self, amount: int) -> List[int]:
        withdrawn = [0]*5
        for i, cash, val in zip(count(4, -1), self.cash[::-1], self.values[::-1]):
            used = min(cash, amount//val)
            amount -= used*val
            withdrawn[i] = used
        if amount == 0:
            self.deposit([-x for x in withdrawn])
            return withdrawn
        return [-1]
```

## 2242. Maximum Score of a Node Sequence

### Solution 1: Undirected graph + hash table

Using the idea of having two fixed nodes and then each fixed node has a single neighbor, want to find the 
maximum combination, to do this need to score the 3 largest valued neighbors for each node

```py
class Solution:
    def maximumScore(self, scores: List[int], edges: List[List[int]]) -> int:
        n = len(scores) # nodes [0,n-1]
        graph_lst = [[] for _ in range(n)]
        for u, v in edges:
            graph_lst[u].append((scores[v], v))
            graph_lst[v].append((scores[u], u))
        for i in range(n):
            graph_lst[i] = nlargest(3, graph_lst[i])
        max_score = -1
        for u, v in edges:
            for nuscore, nu in graph_lst[u]:
                for nvscore, nv in graph_lst[v]:
                    if nu!=nv and nu!=v and nv!=u:
                        max_score = max(max_score, scores[u]+scores[v]+nvscore+nuscore)
        return max_score
```

# Leetcode Biweekly Contest 77

## Summary

## 2255. Count Prefixes of a Given String

### Solution 1: check prefixes

```py
class Solution:
    def countPrefixes(self, words: List[str], s: str) -> int:
        return sum(1 for word in words if s[:len(word)] == word)
```

## 2256. Minimum Average Difference

### Solution 1: prefix sum and suffix sum

```py
class Solution:
    def minimumAverageDifference(self, nums: List[int]) -> int:
        n = len(nums)
        psum, ssum = 0, sum(nums)
        minVal, index = inf, 0
        for i, num in enumerate(nums):
            ssum -= num
            psum += num
            pavg = psum//(i+1)
            savg = ssum//(n-i-1) if n-i-1 > 0 else 0
            if (avgDiff := abs(pavg-savg)) < minVal:
                minVal = avgDiff
                index = i
        return index
```

## 2257. Count Unguarded Cells in the Grid

### Solution 1: 

```py
class Solution:
    def countUnguarded(self, m: int, n: int, guards: List[List[int]], walls: List[List[int]]) -> int:
        # m is rows, n is cols
        grid = [['U']*n for _ in range(m)] # 0 represents unguarded
        for r, c in walls:
            grid[r][c] = 'w' # 3 represents wall block
        for r, c in guards:
            grid[r][c] = 'G' # 2 represents guard
            for row in range(r+1,m):
                if grid[row][c] in ('E', 'G', 'w'): break
                grid[row][c] = 'E'
            for row in range(r-1,-1,-1):
                if grid[row][c] in ('W', 'G', 'w'): break
                grid[row][c] = 'W'
            for col in range(c+1,n):
                if grid[r][col] in ('S', 'G', 'w'): break
                grid[r][col] = 'S'
            for col in range(c-1,-1,-1):
                if grid[r][col] in ('N', 'G', 'w'): break
                grid[r][col] = 'N'
        cnt = 0
        for r, c in product(range(m), range(n)):
            cnt += (grid[r][c]=='U')
        return cnt
```

## 2258. Escape the Spreading Fire

### Solution 1: multisource bfs for fire + binary search for start time

```py
class Solution:
    def maximumMinutes(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        # can escape at this start time for the person
        def canEscape(start_time):
            queue = deque([(0,0,start_time)])
            visited = [[0]*C for _ in range(R)]
            visited[0][0] = 1
            while queue:
                row, col, time = queue.popleft()
                if row==R-1 and col==C-1: return True
                for nr, nc in [(row+1,col),(row-1,col),(row,col+1),(row,col-1)]:
                    if not in_boundary(nr,nc) or time+1 >= fire_time[R-1][C-1] or visited[nr][nc]: continue
                    queue.append((nr,nc,time+1))
                    visited[nr][nc] = 1
            return False
        # BUILD THE MULTISORCE BFS FOR FIRE TO BUILD THE TIMES
        fire_time = [[-1]*C for _ in range(R)]
        queue = deque()
        in_boundary = lambda r, c: 0<=r<R and 0<=c<C
        for r, c in product(range(R), range(C)):
            if grid[r][c] == 1:
                queue.append((r,c, 0))
                fire_time[r][c] = 0
        while queue:
            row, col, time = queue.popleft()
            for nr, nc in [(row+1,col),(row-1,col),(row,col+1),(row,col-1)]:
                if not in_boundary(nr,nc) or grid[nr][nc] == 2 or fire_time[nr][nc] != -1: continue
                fire_time[nr][nc] = time+1
                queue.append((nr,nc,time+1))
        left, right = 0, 100000
        print(fire_time)
        if fire_time[R-1][C-1] == -1: return 1000000000
        if not canEscape(0): return -1
        while left < right:
            mid = (left+right+1)>>1
            # print(left, mid, right)
            if canEscape(mid):
                left = mid
            else:
                right = mid-1
        
        return left
```

### Solution 2: BFS + compute distance to the safehouse for fire and person + treat edge case when both reach at same time but can reach coming from different direction

```py
class Solution:
    def maximumMinutes(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        in_boundary = lambda r, c: 0<=r<R and 0<=c<C
        def bfs(queue):
            dist = {node: 0 for node in queue}
            while queue:
                r, c = queue.popleft()
                for nr, nc in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
                    if not in_boundary(nr,nc) or grid[nr][nc] == 2 or (nr,nc) in dist: continue
                    dist[(nr,nc)] = dist[(r,c)] + 1
                    queue.append((nr,nc))
            return dist
        queue = deque()
        for r, c in product(range(R), range(C)):
            if grid[r][c] == 1:
                queue.append((r,c))
        dist_fire = bfs(queue)
        dist_person = bfs(deque([(0,0)]))
        
        if (R-1,C-1) not in dist_person: return -1
        if (R-1,C-1) not in dist_fire: return 10**9
        
        def time(r,c):
            return dist_fire[(r,c)] - dist_person[(r,c)]
        t = time(R-1,C-1)
        if grid[-1][-2] !=2 and grid[-2][-1] != 2 and max(time(R-1,C-2), time(R-2,C-1)) > t:
            t+=1
        return max(t - 1,-1)
```

# Leetcode Biweekly Contest 100

## 2591. Distribute Money to Maximum Children

### Solution 1:  math + division theory

d = nq + r, where money = 7n + r in this case. 

give all children 1 dollar, if not enough return -1 else use this and try to give 7 more to get it to 8 dollars for everyone. 

sometimes there is extra giving, was able to give to more children than need to, then you can put that into the remainder.  And basically if the remainder is greater than 0 that means you cannot give to more than children - 1, cause the last child will need to push all that extra remainder onto. 

But there is one more special case which is when the remainder = 3, then this only applies when technically you have n = children - 1, cause then you can't give 3 to the last child but need to split it between two so that means you will need to subtract 1 more from children.

```py
class Solution:
    def distMoney(self, money: int, children: int) -> int:
        money -= children
        if money < 0: return -1
        n, r = divmod(money, 7)
        extra_giving = max(0, n - children)
        r += 7*extra_giving
        return min(children - (r > 0) - (n < children and r == 3), n)
```

## 2592. Maximize Greatness of an Array

### Solution 1:  sort + two pointers

sort array and use left to point to cur smallest element that has not have found another element in the array that is greater than it.  Once that happens increment left and left pointer will be the number of elements that were able to have a permutated element from same array that is greater than it. 

```py
class Solution:
    def maximizeGreatness(self, nums: List[int]) -> int:
        nums.sort()
        left = 0
        for right in range(len(nums)):
            if nums[right] > nums[left]:
                left += 1
        return left
```

## 2593. Find Score of an Array After Marking All Elements

### Solution 1:  sort + hash table

just have to track the marked elements, and do it in sorted order from smallest to largest. 

```py
class Solution:
    def findScore(self, nums: List[int]) -> int:
        score = 0
        marked = [0]*(len(nums) + 1)
        for i, num in sorted(enumerate(nums), key = lambda pair: pair[1]):
            if marked[i]: continue
            marked[i] = marked[i - 1] = marked[i + 1] = 1
            score += num
        return score
```

## 2594. Minimum Time to Repair Cars

### Solution 1:  greedy binary search 

This became way faster after adding the Counter cause now it puts ranks into buckets, so decreases the size of iteration sometimes, obviously worse case is when every single mechanic has a different rank.

```py
class Solution:
    def repairCars(self, ranks: List[int], cars: int) -> int:
        f = lambda time, rank: math.isqrt(time//rank)
        counts = Counter(ranks)
        left, right = 0, min(counts)*cars*cars
        def possible(time):
            repaired = 0
            for rank, cnt in counts.items():
                repaired += cnt*f(time, rank)
                if repaired >= cars: return True
            return False
        while left < right:
            mid = (left + right) >> 1
            if possible(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

### Solution 2:  binary search with bisect left + custom comparator 

This will check for each time in range of 0 to min(counts)*cars*cars cand check based on the key which will iterate through all the counts of ranks and find the total number of cars repaired and compared it to the cars variable, 
so if cars <= repaired_cars it will return True

```py
class Solution:
    def repairCars(self, ranks: List[int], cars: int) -> int:
        f = lambda time, rank: math.isqrt(time//rank)
        counts = Counter(ranks)
        return bisect.bisect_left(range(0, min(counts)*cars*cars), cars, key = lambda time: sum(cnt*f(time, rank) for rank, cnt in counts.items()))
```

This one is a little more clear, we are going to say when repaired cars is greater than equal to cars then return True
I want the first instance of True, 

cause we will have something like this, monotonic array of repaired cars as time increases

[1,1,2,2,3,3,4,5,6,7,8,9]
FFFFTTTTT

first T is when the first instance that repaired cars is greater than or equal to cars, so can repair all cars within time with these mechanics

```py
class Solution:
    def repairCars(self, ranks: List[int], cars: int) -> int:
        f = lambda time, rank: math.isqrt(time//rank)
        counts = Counter(ranks)
        return bisect.bisect_left(range(0, min(counts)*cars*cars), True, key = lambda time: sum(cnt*f(time, rank) for rank, cnt in counts.items()) >= cars)
```

# Leetcode Biweekly Contest 101

## 2605. Form Smallest Number From Two Digit Arrays

### Solution 1:

```py
class Solution:
    def minNumber(self, nums1: List[int], nums2: List[int]) -> int:
        min1 = min2 = math.inf
        for dig in map(int, string.digits):
            if dig in nums1 and dig in nums2:
                return dig
            if dig in nums1:
                min1 = min(min1, dig)
            if dig in nums2:
                min2 = min(min2, dig)
        return min1*10 + min2 if min1 < min2 else min2*10 + min1
```

## 2606. Find the Substring With Maximum Cost

### Solution 1:  dp + kadanes

This is similar to kadanes' algo to get the maximum cost of a subarray. 

```py
class Solution:
    def maximumCostSubstring(self, s: str, chars: str, vals: List[int]) -> int:
        values = list(range(1, 27))
        unicode = lambda ch: ord(ch) - ord('a')
        for ch, v in zip(chars, vals):
            i = unicode(ch)
            values[i] = v
        res = cur = 0
        for ch in s:
            i = unicode(ch)
            cur = max(0, cur + values[i])
            res = max(res, cur)
        return res
```

## 2608. Shortest Cycle in a Graph

### Solution 1: dfs + depth array

```py
class Solution:
    def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        res = math.inf
        depth = [0]*n
        def dfs(node, parent):
            nonlocal res
            for nei in adj_list[node]:
                if nei == parent: continue
                if depth[nei]: 
                    res = min(res, abs(depth[node] - depth[nei]) + 1)
                    continue
                depth[nei] = depth[node] + 1
                dfs(nei, node)
        for i in range(n):
            depth[i] = 1
            dfs(i, -1)
            depth = [0]*n
        return res if res < math.inf else -1
```

### Solution 2:  bfs + early termination + dist array 

can use the distance from root to know if it is parent or not, cause if distance is smaller, it is parent

total distance is distance from root to nei and node + 1 to get the size of the smallest cycle found, which is the first cycle. 
Although sometimes it counts the value as too large cause it is not always getting size of cycle if the root node does not belong to a cycle. 
But since minimizing when treating element in the cycle as root that will be the shortest cycle anyway it still works. 

```py
class Solution:
    def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        def bfs(root):
            # dist from root node
            dist = [-1]*n
            dist[root] = 0
            queue = deque([root])
            while queue:
                node = queue.popleft()
                for nei in adj_list[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        queue.append(nei)
                    elif dist[nei] >= dist[node]:
                        return dist[nei] + dist[node] + 1
            return math.inf
       
        res = min(map(bfs, range(n)))
        return res if res < math.inf else -1
```

## 2607. Make K-Subarray Sums Equal

### Solution 1:  gcd + cycle + sort

This one is weird, still don't have a fully understanding of it. Why does the gcd work I can't prove it. But you can also solve this
by creating a visited array and just finding the cycles that way. 

basic pattern is that you need certain elements equal such as n = 4, k = 2
a(0) + a(1) = a(1) + a(2) = ...

so basically need a(i) = a(i + k), so this can be solved with a loop and a secondardy loop that goes through increments of k to build the cycle 
and just store visited for the outer loop so don't redo. 

```py
class Solution:
    def makeSubKSumEqual(self, arr: List[int], k: int) -> int:
        n = len(arr)
        dist = gcd(n, k)
        res = 0
        for i in range(dist):
            cycle = sorted([arr[j] for j in range(i, n, dist)])
            median = cycle[len(cycle)//2]
            res += sum([abs(v - median) for v in cycle])
        return res
```

# Leetcode Biweekly Contest 102

## 2639. Find the Width of Columns of a Grid

### Solution 1:  loop + string

```py
class Solution:
    def findColumnWidth(self, grid: List[List[int]]) -> List[int]:
        R, C = len(grid), len(grid[0])
        ans = [0]*C
        for r, c in product(range(R), range(C)):
            ans[c] = max(ans[c], len(str(grid[r][c])))
        return ans
```

## 2640. Find the Score of All Prefixes of an Array

### Solution 1:  prefix sum and max + accumulate

```py
class Solution:
    def findPrefixScore(self, nums: List[int]) -> List[int]:
        n = len(nums)
        pmax = 0
        conver = [0]*n
        for i, num in enumerate(nums):
            pmax = max(pmax, num)
            conver[i] = num + pmax
        return accumulate(conver)
```

```py
class Solution:
    def findPrefixScore(self, nums: List[int]) -> List[int]:
        return accumulate([num + pmax for num, pmax in zip(nums, accumulate(nums, max))])
```

## 2641. Cousins in Binary Tree II

### Solution 1:  2 dfs + first dfs to compute level sum + second dfs to compute cousins value by level_sum - children values

```py
class Solution:
    def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        level_sum = Counter()
        def dfs1(node, depth):
            for child in filter(None, (node.left, node.right)):
                level_sum[depth] += child.val
                dfs1(child, depth + 1)
        dfs1(root, 1)
        def dfs2(node, root, depth):
            children_val = (root.left.val if root.left else 0) + (root.right.val if root.right else 0)
            cousins_val = level_sum[depth + 1] - children_val
            if root.left:
                node.left = TreeNode(cousins_val)
                dfs2(node.left, root.left, depth + 1)
            if root.right:
                node.right = TreeNode(cousins_val)
                dfs2(node.right, root.right, depth + 1)
            return node
        return dfs2(TreeNode(0), root, 0)
```

## 2642. Design Graph With Shortest Path Calculator

### Solution 1:  dijkstra

```py
class Graph:

    def __init__(self, n: int, edges: List[List[int]]):
        self.adj_list = [[] for _ in range(n)]
        for u, v, w in edges:
            self.adj_list[u].append((v, w))
        self.n = n
        
    def addEdge(self, edge: List[int]) -> None:
        u, v, w = edge
        self.adj_list[u].append((v, w))
        
    def shortestPath(self, node1: int, node2: int) -> int:
        minheap = [(0, node1)]
        min_dist = [math.inf]*self.n
        min_dist[node1] = 0
        while minheap:
            cost, node = heappop(minheap)
            if cost > min_dist[node]: continue
            if node == node2: return cost
            for nei, wei in self.adj_list[node]:
                ncost = cost + wei
                if ncost < min_dist[nei]:
                    heappush(minheap, (ncost, nei))
                    min_dist[nei] = ncost
        return -1
```

# Leetcode Biweekly Contest 103

## 2656. Maximum Sum With Exactly K Elements 

### Solution 1:  max + math

Use the summation of natural numbers from 1 + 2 + ... + k - 1

```py
class Solution:
    def maximizeSum(self, nums: List[int], k: int) -> int:
        return k*max(nums) + (k - 1)*k//2
```

## 2657. Find the Prefix Common Array of Two Arrays

### Solution 1: prefix count + frequency array

```py
class Solution:
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        n = len(A)
        counts = [0]*(n + 1)
        res = [0]*n
        prefix_count = 0
        for i in range(n):
            counts[A[i]] += 1
            prefix_count += counts[A[i]] == 2
            counts[B[i]] += 1
            prefix_count += counts[B[i]] == 2
            res[i] = prefix_count
        return res
```

## 2658. Maximum Number of Fish in a Grid

### Solution 1:  dfs + constant space memoization + modify grid in-place

```py
class Solution:
    def findMaxFish(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        def dfs(r, c):
            stack = [(r, c, grid[r][c])]
            res = 0
            grid[r][c] = 0
            while stack:
                r, c, val = stack.pop()
                res += val
                for nr, nc in [(r + 1, c), (r - 1, c), (r, c - 1), (r, c + 1)]:
                    if not in_bounds(nr, nc) or grid[nr][nc] == 0: continue
                    stack.append((nr, nc, grid[nr][nc]))
                    grid[nr][nc] = 0
            return res
        result = 0
        for r, c in product(range(R), range(C)):
            if grid[r][c] == 0: continue
            result = max(result, bfs(r, c))
        return result
```

## 2659. Make Array Empty

### Solution 1:  modular arithmetic + bit(fenwick tree) + pointer + sort

Use a pointer to track current index and sort the order of index to visit, so then go and visit the index in order.  So you can find the distance between previous and current index in the array.  There are two different cases for when you wrap around and when you don't wrap around in the array. Then need to use a binary indexed tree for range sum queries and range updates that are fast.  It is just going to store the index that have already been removed, because those should be skipped and not counted as an operation.  So you store the count in the bit. 

```py
class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def __repr__(self):
        return f"array: {self.sums}"

class Solution:
    def countOperationsToEmptyArray(self, nums: List[int]) -> int:
        n = len(nums)
        index = sorted(list(range(n)), key = lambda i: nums[i])
        fenwick = FenwickTree(n)
        i = res = 0
        for idx in index:
            delta = 0
            if idx >= i:
                delta = idx - i + 1
                left = fenwick.query(i)
                i += delta
                right = fenwick.query(i)
                i %= n
                seg_sum = right - left
                delta -= seg_sum
            else:
                delta = n - i + idx + 1
                left = fenwick.query(i)
                right = fenwick.query(n)
                left_seg_sum = right - left
                i += delta
                i %= n
                right_seg_sum = fenwick.query(i)
                delta = delta - left_seg_sum - right_seg_sum
            res += delta
            fenwick.update(idx + 1, 1)
        return res
```

### Solution 2:  sort + position + simulation

The idea is that you are simulation how many rounds it takes to remove all the elements.  In this manner each round will have the number of operations be the remaining elements.  So just need to iterate through sorted nums and check if position is less than previous to indicate that it has began a round starting from the beginning that will not need to add the remaining elements to the total operations required.

```py
class Solution:
    def countOperationsToEmptyArray(self, nums: List[int]) -> int:
        n = len(nums)
        pos = {num: i for i, num in enumerate(nums)}
        res = n
        nums.sort()
        for i in range(1, n):
            if pos[nums[i]] < pos[nums[i - 1]]:
                res += n - i
        return res
```

# Leetcode Biweekly Contest 104

## 2678. Number of Senior Citizens

### Solution 1:  string slicing

```py
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        return sum(1 for info in details if int(info[11:13]) > 60)
```

## 2679. Sum in a Matrix

### Solution 1:  sort rows + transpose matrix

By sorting the rows in reverse guarantee largest elements are first, then by taking transpose, each column will be turned into rows in new matrix. Then we can sum the largest element in each row. Because that corresponds to largest value in each column

```py
class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        for row in nums:
            row.sort(reverse = True)
        def transpose_matrix(matrix):
            return list(map(list, zip(*matrix)))
        mat = transpose_matrix(nums)
        return sum(max(row) for row in mat)
```

## 2680. Maximum OR

### Solution 1:  frequency array

```py
class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        freq = [0]*45
        res = 0
        for num in nums:
            for j in range(32):
                if (num>>j)&1:
                    freq[j] += 1
        for num in nums:
            cur = 0
            for j in range(45):
                if (num>>j)&1:
                    freq[j] -= 1
                    freq[j + k] += 1
                if freq[j] > 0:
                    cur |= (1 << j)
            res = max(res, cur)
            for j in range(45):
                if (num>>j)&1:
                    freq[j] += 1
                    freq[j + k] -= 1
        return res
```

### Solution 2:  prefix and suffix or sum + bit manipulation

Consider current element and multiplying by 2^k or shifting bits to the left by k.  Then or that with the pref_or and suffix_or

```py
class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pref_or, suf_or = 0, [0]*(n+1)
        for i in range(n - 1, -1, -1):
            suf_or[i] = suf_or[i+1] | nums[i]
        res = 0
        for i in range(n):
            val = nums[i] << k
            res = max(res, pref_or | val | suf_or[i+1])
            pref_or |= nums[i]
        return res
```

## 2681. Power of Heroes

### Solution 1:  sort + math + deque + prefix sum

![derivation](images/power_of_heroes.png)

```py
class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        res = 0
        n = len(nums)
        mod = int(1e9) + 7
        q = deque()
        psum1 = psum2 = 0
        for num in nums:
            psum1 = (psum1 + num)%mod
            q.append(num)
            if len(q) > 2:
                x = q.popleft()
                psum1 -= x
                psum2 += x
            psum2 = (psum2*2)%mod
            res = (res + ((psum1 + psum2)*pow(num, 2, mod))%mod)%mod
        return res
```

# Leetcode Biweekly Contest 105

## 2706. Buy Two Chocolates

### Solution 1:  max

```py
class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        return money - prices[0] - prices[1] if money - prices[0] - prices[1] >= 0 else money
```

## 2707. Extra Characters in a String

### Solution 1:  dynamic programming + O(n^3)

dp[i] = min extra characters for the substring s[0:i] or range [0, i)

```py
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dictionary = set(dictionary)
        dp = [0] + [math.inf]*n
        for j in range(n + 1):
            for i in range(j):
                if s[i:j] in dictionary:
                    dp[j] = min(dp[j], dp[i])
                else:
                    dp[j] = min(dp[j], dp[i]+j-i)
        return dp[-1]
```

```cpp
class Solution {
public:
    int minExtraChar(string s, vector<string>& dictionary) {
        unordered_set<string> seen(dictionary.begin(), dictionary.end());
        int N = s.size();
        vector<int> dp(N + 1, 0);
        for (int i = 0; i < N; i++) {
            dp[i + 1] = dp[i] + 1;
            string cur = "";
            for (int j = i; j >= 0; j--) {
                cur = s[j] + cur;
                if (seen.count(cur)) {
                    dp[i + 1] = min(dp[i + 1], dp[i - cur.size() + 1]);
                }
            }
        }
        return dp[N];
    }
};
```

## 2708. Maximum Strength of a Group

### Solution 1:  sort + O(nlogn)

take even number of negative integers because they cancel to positive, take all positive integers.
only reason to take odd number of negative integers is if there is a single element in array and it is negative. 

```py
class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1: return nums[0]
        nums.sort()
        pos_i = bisect.bisect_right(nums, 0)
        neg_i = bisect.bisect_right(nums, -1)
        res = 0
        if neg_i & 1:
            neg_i -= 1
        for i in range(neg_i):
            if res == 0: res = 1
            res *= nums[i]
        for i in range(pos_i, n):
            if res == 0: res = 1
            res *= nums[i]
        return res
```

## 2709. Greatest Common Divisor Traversal

### Solution 1:  union find + prime sieve

If there is a single connected component, than there is always a traversal sequence through the graph to get all pairs be able to reach each other.  All elements that share a prime factor should be grouped into same connected component, because you can also get from one to the other with gcd > 1.

Therefore, simply merge the elements with common prime factors together, and finally check if they can form a whole large set.

can use the first array to track what was the first index of element that had that prime integer, only need to store one representative integer for each prime, so everything gets merged into that representative integer.

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    """
    returns true if the nodes were not union prior. 
    """
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
        return sum(node == self.find(node) for node in range(len(self.parent)))

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        mx = max(nums)
        def prime_sieve(lim):
            sieve,primes = [set() for _ in range(lim)], []
            for integer in range(2,lim):
                if not len(sieve[integer]):
                    primes.append(integer)
                    for possibly_divisible_integer in range(integer,lim,integer):
                        current_integer = possibly_divisible_integer
                        while not current_integer%integer:
                            sieve[possibly_divisible_integer].add(integer)
                            current_integer //= integer
            return sieve
        sieve = prime_sieve(mx + 1)
        dsu = UnionFind(n)
        first = [-1] * (mx + 1)
        for i, num in enumerate(nums):
            for prime in sieve[num]:
                if first[prime] != -1: dsu.union(first[prime], i)
                else: first[prime] = i
        return dsu.root_count == 1
```

### Solution 2:  online prime factorization + union find

This is slower than method above most likely and leetcode confirmed that statement given their test cases.

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    """
    returns true if the nodes were not union prior. 
    """
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
        return sum(node == self.find(node) for node in range(len(self.parent)))

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        last = [-1] * (1 + max(nums))
        dsu = UnionFind(n)
        for i, num in enumerate(nums):
            for prime in range(2, num):
                if prime*prime > num: break
                if num % prime != 0: continue
                if last[prime] != -1: dsu.union(last[prime], i)
                else: last[prime] = i
                while num % prime == 0: num //= prime
            if num > 1:
                if last[num] != -1: dsu.union(last[num], i)
                else: last[num] = i
        return dsu.root_count == 1
```

# Leetcode Biweekly Contest 106

## 2729. Check if The Number is Fascinating

### Solution 1: string + set

```py
class Solution:
    def isFascinating(self, n: int) -> bool:
        num = str(n) + str(2 * n) + str(3 * n)
        return len(set(num)) == 9 and len(num) == 9 and '0' not in set(num)
```

## 2730. Find the Longest Semi-Repetitive Substring

### Solution 1:  brute force + loops + substrings + O(n^3)

```py
class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        res = 0
        for i in range(n):
            for j in range(i + 1, n + 1):
                pairs = 0
                for k in range(i + 1, j):
                    if s[k - 1] == s[k]:
                        pairs += 1
                if pairs <= 1:
                    res = max(res, j - i)
        return res
```

## 2731. Movement of Robots

### Solution 1:  math + prefix sum + ignore collisions

This is like ants on stick problem. 

![image](images/movement_of_robots.PNG)

```py
class Solution:
    def sumDistance(self, nums: List[int], s: str, d: int) -> int:
        n = len(nums)
        mod = int(1e9) + 7
        nums = sorted([nums[i] + (d if s[i] == 'R' else -d) for i in range(n)])
        psum = add_sum = sub_sum = 0
        for i in range(n):
            print(i, ~i)
            add_sum += nums[~i] % mod
            sub_sum += nums[i] % mod
            psum += add_sum % mod
            psum = (psum - sub_sum + mod) % mod
        return psum
```

## 2732. Find a Good Subset of the Matrix

### Solution 1:  greedy + bitmask

If a good subset exists, than there will also exist a good subset with 1 or 2 rows.  So just need to check for good subsets of length 1 if all columns are 0, and subsets of length 2 are good by using bitmask, that is there cannot be two 1s in both columns, must be at most 1 1 in a column

```py
class Solution:
    def goodSubsetofBinaryMatrix(self, grid: List[List[int]]) -> List[int]:
        R, C = len(grid), len(grid[0])
        for r, row in enumerate(grid):
            if sum(row) == 0: return [r]
        states = {}
        for r, row in enumerate(grid):
            mask = sum(1 << c for c, val in enumerate(row) if val)
            for pmask in range(1 << C):
                if pmask & mask: continue
                if pmask in states:
                    return [states[pmask], r]
            states[mask] = r
        return []
```

# Leetcode Biweekly Contest 107

## 2744. Find Maximum Number of String Pairs

### Solution 1:  brute force

```py
class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        n = len(words)
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                if words[i] == words[j][::-1]:
                    res += 1
                    break
        return res
```

## 2745. Construct the Longest New String

### Solution 1:  math

derived these by realizing legal pairs were (xy, yx, zx, yz, zz)
1. if x > y, can take all the z strings, take one extra x
zz...zzxyxy...xyx
2. if y > x, can take all the z strings, take on extra y
yxyx....yxyzz...zz
3. if x == y, take everything x + y + z

```py
class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        a = min(x, y)
        extra = min(1, max(x, y) - a)
        res = 2 * a + extra + z
        return 2 * res
```

## 2746. Decremental String Concatenation

### Solution 1:  dynamic programming + minimize the length of the concatenated string

two recurrence relations, and the dp states for 0...i words are already solved, and the only thing that matters is the
(first_character, last_character) that makes up that concatenated string, and store the length, cause want to minimize on that. 

So updates will be something like dp[first][last] = min(dp[first][last], ....)

read code explains logic well

```py
class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = {(words[0][0], words[0][-1]): len(words[0])}
        for i in range(1, n):
            ndp = defaultdict(lambda: math.inf)
            fc, lc = words[i][0], words[i][-1]
            for (s, e), l in dp.items():
                ndp[(s, lc)] = min(ndp[(s, lc)], l + len(words[i]) - (1 if e == fc else 0))
                ndp[(fc, e)] = min(ndp[(fc, e)], l + len(words[i]) - (1 if s == lc else 0))
            dp = ndp
        return min(dp.values())
```

## 2747. Count Zero Request Servers

### Solution 1:  offline queries + sort + two pointers + sliding window of size x + frequency counter

sort the queries and the logs based on time. 
Then take two pointers left, right
And move the right pointer up to the current queries[i]
And move the left pointer up to less than queries[i] - x 

track frequency of each server, and update the cnt appropriately
cnt represents the number of server that have received a server request in the current window
So the answer will be total number of servers - cnt will give server with 0 requests in the current query window.

```py
class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        nlogs, m = len(logs), len(queries)
        cnt = 0
        freq = Counter()
        queries = sorted([(v, i) for i, v in enumerate(queries)])
        ans = [0] * m
        left = right = 0
        logs.sort(key = lambda x: x[1])
        for v, i in queries:
            while right < nlogs and logs[right][1] <= v:
                server = logs[right][0]
                freq[server] += 1
                if freq[server] == 1:
                    cnt += 1
                right += 1
            while left < nlogs and logs[left][1] < v - x:
                server = logs[left][0]
                freq[server] -= 1
                if freq[server] == 0:
                    cnt -= 1
                left += 1
            ans[i] = n - cnt
        return ans
```

# Leetcode Biweekly Contest 108

## 2765. Longest Alternating Subarray

### Solution 1:  sliding window

```py
class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        res = -1
        left = 0
        n = len(nums)
        diff = [nums[i] - nums[i - 1] for i in range(1, n)]
        for right in range(n - 1):
            while left < right and diff[left] != 1:
                left += 1
            delta = right - left
            if ((delta & 1) and diff[right] == -1) or (delta % 2 == 0 and diff[right] == 1):
                res = max(res, delta + 2)
            else:
                left = right
        return res
```

## 2766. Relocate Marbles

### Solution 1:  counter

```py
class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        locations = Counter(nums)
        for u, v in zip(moveFrom, moveTo):
            cnt = locations[u]
            locations[u] -= cnt
            locations[v] += cnt
        return sorted([k for k, v in locations.items() if v > 0])
```

## 2767. Partition String Into Minimum Beautiful Substrings

### Solution 1:  dfs + backtrack

```py
class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        fives = {1, 5, 25, 125, 625, 3125, 15625}
        substrings = []
        def backtrack(i):
            if i == n: return 0
            res = math.inf
            if s[i] == '0': return res
            for j in range(i, n):
                cand = int(s[i:j+1], 2)
                if cand in fives:
                    substrings.append(cand)
                    res = min(res, backtrack(j+1) + 1)
                    substrings.pop()
            return res
        res = backtrack(0)
        return res if res < math.inf else -1
```

### Solution 2:  dynamic programming + O(n^2)

dp[i] = minimum number of valid partitions ending at character s[i - 1]
base case is dp[0] for empty string and set to 0 partitions

```py
class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        fives = {1, 5, 25, 125, 625, 3125, 15625}
        dp = [0] + [math.inf] * n
        for i in range(1, n + 1):
            for j in range(i):
                if s[j] == '0': continue
                cand = int(s[j:i], 2)
                if cand not in fives: continue
                dp[i] = min(dp[i], dp[j] + 1)
        return dp[-1] if dp[-1] != math.inf else -1
```

## 2768. Number of Black Blocks

### Solution 1:  hash table + counters

For each black rock, add it to all the possibly 4 submatrices it can belong within.  

```py
class Solution:
    def countBlackBlocks(self, R: int, C: int, coordinates: List[List[int]]) -> List[int]:
        in_bounds = lambda r, c: 0 <= r < R - 1 and 0 <= c < C - 1
        neighborhood = lambda r, c: [(r - 1, c), (r - 1, c - 1), (r, c - 1), (r, c)]
        black_counter = Counter()
        for r, c in coordinates:
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc): continue
                cell = nc + nr * C
                black_counter[cell] += 1
        counts = [0] * 5
        for cnt in black_counter.values():
            counts[cnt] += 1
        counts[0] = (R - 1) * (C - 1) - sum(counts)
        return counts
```

# Leetcode Biweekly Contest 109

## 6930. Check if Array is Good

### Solution 1:  counts

```py
class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        cnt = [0] * (n + 1)
        for num in nums:
            cnt[num] += 1
        return all(cnt[i] == 1 for i in range(1, n)) and cnt[n] == 2
```

## 6926. Sort Vowels in a String

### Solution 1:  sort + string

```py
class Solution:
    def sortVowels(self, s: str) -> str:
        indices, vows = [], []
        vowels = "AEIOUaeiou"
        for i, c in enumerate(s):
            if c in vowels:
                indices.append(i)
                vows.append(c)
        vows.sort()
        res = list(s)
        for i, v in zip(indices, vows):
            res[i] = v
        return ''.join(res)
```

## 6931. Visit Array Positions to Maximize Score

### Solution 1:  dynamic programming

just store the max score so far considering the parity of current element, it has two previous it can come from.

```py
class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        if nums[0] & 1:
            odd = nums[0]
            even = -math.inf
        else:
            even = nums[0]
            odd = -math.inf
        for num in nums[1:]:
            if num & 1:
                odd = max(odd + num, even + num - x)
            else:
                even = max(even + num, odd + num - x)
        return max(even, odd)
```

## 6922. Ways to Express an Integer as Sum of Powers

### Solution 1:  dynamic programming + counter

```py
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        mod = int(1e9) + 7
        dp = Counter({0: 1})
        for i in range(1, n + 1):
            v = pow(i, x)
            if v > n: break
            ndp = dp.copy()
            for k, cnt in dp.items():
                if k + v > n: continue
                ndp[k + v] += cnt
                ndp[k + v] %= mod
            dp = ndp
        return dp[n]
```

# Leetcode Biweekly Contest 110

## 2806. Account Balance After Rounded Purchase

### Solution 1:  brute force

```py
class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        best, diff = 0, math.inf
        for i in range(0, 101, 10):
            if abs(purchaseAmount - i) <= diff:
                best = i
                diff = abs(purchaseAmount - i)
        return 100 - best
```

## 2807. Insert Greatest Common Divisors in Linked List

### Solution 1:  linked list + inserting nodes + gcd

```py
class Solution:
    def insertGreatestCommonDivisors(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        while cur.next:
            nxt = cur.next
            cur.next = ListNode(math.gcd(cur.val, nxt.val))
            cur.next.next = nxt
            cur = nxt
        return head
```

## 2808. Minimum Seconds to Equalize a Circular Array

### Solution 1: dictionary + maximize

you just need to store the last index for when last a value appeared, and then you need to calculate the distance between current occurrence and last index of that integer. The number of seconds required to fill in everything between it with it's value will be (dist + 1) // 2. You need to maximize this value for each integer.

proof is 
x _ _ _ x, 
it will take 2 seconds obviously to fill in 3 slots between an integer
1st second x x _ x x
2nd second x x x x x
so you just need to know the number of slots that need to be changed to value x.  Or think of it like yeah, but at each second they can move to right and left, so iti s just dividing by 2. 

```py
class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        last_index = {}
        time = Counter()
        for i in range(n):
            last_index[nums[i]] = i
        for i in range(n):
            dist = (i - last_index[nums[i]] - 1) % n
            last_index[nums[i]] = i
            delta = (dist + 1) // 2
            time[nums[i]] = max(time[nums[i]], delta)
        return min(time.values())
```

## 2809. Minimum Time to Make Array Sum At Most x

### Solution 1:  greedy + sort + dynamic programming + exchange argument

dp[i][j] = maximum value for the first i elements and j operations

To really understand the greedy sorting part you can use exchange argument to prove that you just need to sort nums2 and that anytime you picking the same element it is always optimal to pick it later so that it has a larger multiplier

![image](images/minimum_time_to_make_array_sum_at_most_x.png)

```py
class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        nums = sorted([(x1, x2) for x1, x2 in zip(nums1, nums2)], key = lambda pair: pair[1])
        s1, s2 = sum(nums1), sum(nums2)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i, j in product(range(n), repeat = 2):
            # j + 1 operation
            dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i][j] + nums[i][0] + (j + 1) * nums[i][1])
        for i in range(n + 1):
            if s1 + s2 * i - dp[n][i] <= x: return i
        return - 1
```

# Leetcode BiWeekly  Contest 111

## 2824. Count Pairs Whose Sum is Less than Target

### Solution 1: 

```py
class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        res = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                res += nums[i] + nums[j] < target
        return res
```

## 2825. Make String a Subsequence Using Cyclic Increments

### Solution 1: 

```py
class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n1, n2 = len(str1), len(str2)
        unicode = lambda ch: ord(ch) - ord('a')
        i = 0
        for ch in str2:
            while i < n1 and str1[i] != ch and chr(((unicode(str1[i]) + 1) % 26) + ord('a')) != ch:
                i += 1
            if i == n1: return False
            i += 1
        return True
```

## 2826. Sorting Three Groups

### Solution 1: 

```py
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        psum = [[0] * 4 for _ in range(n + 1)]
        for i in range(n):
            psum[i + 1] = psum[i].copy()
            psum[i + 1][nums[i]] += 1
        res = n
        for i in range(n + 1):
            nones = psum[i][2] + psum[i][3]
            for j in range(i, n + 1):
                ntwos = psum[j][1] + psum[j][3] - psum[i][1] - psum[i][3]
                nthrees = psum[-1][1] + psum[-1][2] - psum[j][1] - psum[j][2]
                res = min(res, nones + ntwos + nthrees)
        return res
```

## 2827. Number of Beautiful Integers in the Range

### Solution 1:  digit dp

```py
class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        # (remainder modulo k, even_count - odd_count, tight, zero)
        def solve(upper):
            dp = Counter({(0, 0, 1, 1): 1})
            for d in map(int, upper):
                ndp = Counter()
                for (rem, parity, tight, zero), cnt in dp.items():
                    for dig in range(10 if not tight else d + 1):
                        nrem, ntight, nzero = (rem * 10 + dig) % k, tight and dig == d, zero and dig == 0
                        nparity = parity + (1 if dig % 2 == 0 else -1) if not nzero else 0
                        ndp[(nrem, nparity, ntight, nzero)] += cnt
                dp = ndp
            return sum(dp[(0, 0, t, 0)] for t in range(2))
        return solve(str(high)) - solve(str(low - 1))
```



# Leetcode BiWeekly  Contest 112

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

x1 * 1 * x2 * 1 * x3 * 3

cause if you can do it 3 ways then you can basically 

the thing is math.comb will be greater than 1 only if take less than ffreq[cnt].  So only for the suffix like above.  So this is equivalent to above solution it is just you can use a more general math formula.  I just didn't realize that the combinations would be 1 for each prefix.  And that you can take the frequency of the frequency and that is actually a more useful value in this problem. 

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



# Leetcode BiWeekly  Contest 113

## 2855. Minimum Right Shifts to Sort the Array

### Solution 1: 

```py
class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        i = 1
        while i < n and nums[i - 1] < nums[i]:
            i += 1
        for j in range(i, n):
            if j > i and nums[j] < nums[j - 1]: return -1
            if nums[j] > nums[i - 1]: return -1
        return n - i
```

## 2856. Minimum Array Length After Pair Removals

### Solution 1:  max heap + counter

```py
class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        freq = Counter(nums)
        max_heap = []
        for num, cnt in freq.items():
            heappush(max_heap, (-cnt, num))
        while len(max_heap) > 1:
            cnt_x, x = heappop(max_heap)
            cnt_y, y = heappop(max_heap)
            cnt_x, cnt_y = map(abs, (cnt_x, cnt_y))
            cnt_x -= 1
            cnt_y -= 1
            if cnt_x > 0:
                heappush(max_heap, (-cnt_x, x))
            if cnt_y > 0:
                heappush(max_heap, (-cnt_y, y))
        return sum(abs(cnt) for cnt, _ in max_heap)
```

## 2857. Count Pairs of Points With Distance k

### Solution 1:  bit manipulation + math

```py

```

## 2858. Minimum Edge Reversals So Every Node Is Reachable

### Solution 1:  reroot tree + tree dp

```py
class Solution:
    def minEdgeReversals(self, n: int, edges: List[List[int]]) -> List[int]:
        dp = [0] * n
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append((v, 0))
            adj_list[v].append((u, 1))
        def dfs(node, parent):
            s = 0
            for nei, wei in adj_list[node]:
                if nei == parent: continue
                s += wei
                s += dfs(nei, node)
            dp[node] = s
            return s
        dfs(0, -1)
        ans = [0] * n
        def dfs2(node, parent, psum):
            ans[node] = dp[node] + psum
            for nei, wei in adj_list[node]:
                if nei == parent: continue
                nsum = psum + (wei ^ 1) + dp[node] - dp[nei] - wei
                dfs2(nei, node, nsum)
        dfs2(0, -1, 0)
        return ans
```



# Leetcode BiWeekly  Contest 114

## 2869. Minimum Operations to Collect Elements

### Solution 1:  visited

```py
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        vis = [0] * k
        n = len(nums)
        for i in reversed(range(n)):
            if nums[i] <= k:
                vis[nums[i] - 1] = 1
            if sum(vis) == k: 
                return n - i
        return n
```

## 2870. Minimum Number of Operations to Make Array Empty

### Solution 1:  counter + remainder on division by 3

if remainder = 2, then when you subtract 2 you get remainder = 0 next so for instance it is ceiling of v / 3 
if remainder = 1, then you get 1 -> 2 -> 0

so it will take 1 or 2 operations before the integer will be divisible by 3.  And then you will just take 3 at a time.

```py
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = Counter(nums)
        if any(v == 1 for v in counts.values()): return -1
        return sum(math.ceil(v / 3) for v in counts.values())
```

## 2871. Split Array Into Maximum Number of Subarrays

### Solution 1:  bitwise and operator + greedy

```py
class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        target = reduce(operator.and_, nums)
        if target > 0: return 1
        cur = 0
        res = 0
        for i in range(n):
            cur = cur & nums[i] if cur > 0 else nums[i]
            if cur == 0: res += 1
        return res
```

## 2872. Maximum Number of K-Divisible Components

### Solution 1:  dp on tree + tree + dfs

```py
class Solution:
    def maxKDivisibleComponents(self, n: int, edges: List[List[int]], values: List[int], k: int) -> int:
        dp = [0] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        def dfs(u, par):
            dp[u] = values[u]
            for v in adj[u]:
                if v == par: continue
                dp[u] += dfs(v, u)
            return dp[u]
        dfs(0, -1)
        return sum(1 for x in dp if x % k == 0)
```



# Leetcode BiWeekly  Contest 115

## 2899. Last Visited Integers

### Solution 1:  stack

```py
class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result, stack = [], []
        k = 0
        for word in words:
            if word == "prev":
                k += 1
                result.append(-1 if k > len(stack) else stack[-k])
            else:
                k = 0
                stack.append(int(word))
        return result
```

## 2901. Longest Unequal Adjacent Groups Subsequence II

### Solution 1:  bfs, backtracking, parent arrays, directed graph, topological ordering, indegrees

```py
class Solution:
    def getWordsInLongestSubsequence(self, n: int, words: List[str], groups: List[int]) -> List[str]:
        adj = [[] for _ in range(n)]
        is_edge = lambda i, j: groups[i] != groups[j] and len(words[i]) == len(words[j]) and sum(1 for x, y in zip(words[i], words[j]) if x != y) == 1
        indegrees = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if is_edge(i, j):
                    adj[i].append(j)
                    indegrees[j] += 1
        queue = deque()
        for i in range(n):
            if indegrees[i] == 0: queue.append(i)
        parents = [None] * n
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                indegrees[v] -= 1
                if indegrees[v] == 0: 
                    parents[v] = u
                    queue.append(v)
        path = []
        while u is not None:
            path.append(words[u])
            u = parents[u]
        return reversed(path)
```

## 2902. Count of Sub-Multisets With Bounded Sum

### Solution 1:  unoptimized dynamic programming solution

Count the number of multisets

knapsack dp problem, can have multiple of same item

you have c of item of size a
update is dp[i] += dp[i - a] + dp[i - a * 2] + ... + dp[i - a * c]

You need to improve this update function though, it mentions the idea of sliding window by keeping the sum of dp[i - a] + ... + dp[i - a * c]

```py
class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        mod = int(1e9) + 7
        n = len(nums)
        dp = [0] * (r + 1)
        freq = Counter(nums)
        dp[0] = freq[0] + 1
        for num in freq:
            if num == 0: continue
            for j in range(r, num - 1, -1):
                for k in range(1, freq[num] + 1):
                    if k * num > j: break
                    dp[j] = (dp[j] + dp[j - k * num]) % mod
        res = 0
        for i in range(l, r + 1):
            res = (res + dp[i]) % mod
        return res
```

```py
from collections import Counter 
class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10 ** 9 + 7 
        counter = Counter(nums)
        dp = [0 for _ in range(r + 1)]
        dp[0] = 1 

        for num, freq in counter.items(): 
            for i in range(r, max(r - num, 0), -1): 
                v = sum(dp[i - num * k] for k in range(freq) if i >= num * k)
                for j in range(i, 0, -num):
                    v -= dp[j] 
                    if j >= num * freq: 
                        v += dp[j - num * freq]
                    dp[j] = (dp[j] + v) % MOD

        return (sum(dp[l:])) * (counter[0] + 1) % MOD
```



# Leetcode BiWeekly  Contest 119

## 2956. Find Common Elements Between Two Arrays

### Solution 1:  counter

```py
class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        def count(nums, other):
            counts = [0] * 101
            for v in other:
                counts[v] = 1
            return sum(1 for num in nums if counts[num])
        ans = [count(nums1, nums2), count(nums2, nums1)]
        return ans
```

## 2957. Remove Adjacent Almost-Equal Characters

### Solution 1:  string

```py
class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        word = list(word)
        res = 0
        unicode = lambda ch: ord(ch) - ord('a')
        difference = lambda c1, c2: abs(unicode(c1) - unicode(c2))
        for i in range(1, n):
            if difference(word[i - 1], word[i]) <= 1:
                res += 1
                word[i] = "#"
        return res
```

## 2958. Length of Longest Subarray With at Most K Frequency

### Solution 1:  sliding window, counter

```py
class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left = res = 0
        freq = Counter()
        element = None
        for right in range(n):
            freq[nums[right]] += 1
            if freq[nums[right]] > k:
                element = nums[right]
            while element is not None:
                freq[nums[left]] -= 1
                if element == nums[left]: element = None
                left += 1
            res = max(res, right - left + 1)
        return res
```

## 2959. Number of Possible Sets of Closing Branches

### Solution 1:  bit mask, brute force, enumerate all sets, dijkstra, adjacency matrix

```py
class Solution:
    def numberOfSets(self, n: int, maxDistance: int, roads: List[List[int]]) -> int:
        # remove unecessary edges
        adj_mat = [[math.inf] * n for _ in range(n)]
        for u, v, w in roads:
            adj_mat[u][v] = min(adj_mat[u][v], w)
            adj_mat[v][u] = min(adj_mat[v][u], w)
        # enumerate every possible set of nodes
        def check(mask):
            # all pairs shortest distance
            for src in range(n):
                # finds shortest distance from source node to every other node using dijkstra
                if (mask >> src) & 1: continue # skip nodes that are in the removed set
                min_heap = [(0, src)]
                dist = [math.inf] * n
                dist[src] = 0
                while min_heap:
                    d, u = heappop(min_heap)
                    for v in range(n):
                        if (mask >> v) & 1: 
                            dist[v] = 0 # it is removed
                            continue
                        if v == u or adj_mat[u][v] == math.inf: continue
                        if dist[v] > d + adj_mat[u][v]:
                            dist[v] = d + adj_mat[u][v]
                            heappush(min_heap, (d + adj_mat[u][v], v))
                if any(d > maxDistance for d in dist): return False
            return True
        return sum(check(mask) for mask in range(1 << n))
```



# Leetcode BiWeekly  Contest 119

## 2971. Find Polygon With the Largest Perimeter

### Solution 1:  sort, reverse iteration, prefix sum

```py
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        psum = sum(nums)
        for i in range(n - 1, 1, -1):
            if psum > 2 * nums[i]: return psum
            psum -= nums[i]
        return -1
```

## 

### Solution 1: 

```py

```

## 

### Solution 1:  

```py

```

# Leetcode BiWeekly  Contest 121

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

## 2999. Count the Number of Powerful Integers

### Solution 1:  digit dp

```py
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # states (index, suffix, tight)
        # suppose n1 = 2, n2 = 2 then 2 - 2 = 0
        # suppose n1 = 2, n2 = 3, then 3 - 2 = 1 if i >= n2 - n1
        # then if n1 = 2, n2 = 4,, and 4 - 2 = 2, and i = 2, then you want -(4 - 2) = -2, or -(n2 - i) is the reversed index, cause it will increase as i increases, and move farther to the back of the suffix string.
        # n2 >= n1
        def solve(upper):
            n1, n2 = len(s), len(upper)
            if n2 < n1: return 0
            dp = Counter({(1, 1): 1})
            for i, d in enumerate(map(int, upper)):
                ndp = Counter()
                for (suffix, tight), cnt in dp.items():
                    for dig in range(limit + 1 if not tight else min(limit, d) + 1):
                        nsuffix = suffix and dig == int(s[-(n2 - i)]) if i >= n2 - n1 else suffix
                        ntight = tight and dig == d
                        ndp[(nsuffix, ntight)] += cnt
                dp = ndp
            return dp[(1, 0)] + dp[(1, 1)]
        return solve(str(finish)) - solve(str(start - 1))
```



# Leetcode BiWeekly Contest 122

## 3011. Find if Array Can Be Sorted

### Solution 1:  brute force, sort, bit count

```py
class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        for i in range(1, n):
            for j in range(i, 0, -1):
                if nums[j - 1].bit_count() != nums[j].bit_count(): break
                if nums[j - 1] > nums[j]: nums[j - 1], nums[j] = nums[j], nums[j - 1]
        return nums == sorted(nums)
```

## 3012. Minimize Length of Array Using Operations

### Solution 1:  greedy

```py
class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        def ceil(x, y):
            return (x + y - 1) // y
        freq = Counter(nums)
        min_val = min(nums)
        if any(0 < x % min_val < min_val for x in nums): return 1
        return ceil(freq[min_val], 2)
```

## 3013. Divide an Array Into Subarrays With Minimum Cost II

### Solution 1:  sliding window, max and min heap

```py
class Solution:
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        n = len(nums)
        start = nums[0]
        nums = nums[1:]
        ans = math.inf
        wsum = wcount = 0
        k -= 1
        maxheap, minheap = [], []
        used = [0] * n
        for i in range(n - 1):
            l = i - dist
            if l > 0 and used[l - 1]: # remove outside window
                used[l - 1] = 0
                wsum -= nums[l - 1]
                wcount -= 1
            while wcount < k and minheap: # take from minheap if need more in window
                _, idx = heappop(minheap)
                if idx < l: continue 
                used[idx] = 1
                wsum += nums[idx]
                heappush(maxheap, (-nums[idx], idx))
                wcount += 1
            while maxheap and maxheap[0][1] < l: heappop(maxheap) # remove outsidw window from maxheap
            if wcount < k or nums[i] < -maxheap[0][0]: # add current to maxheap or minheap
                heappush(maxheap, (-nums[i], i))
                wcount += 1
                wsum += nums[i]
                used[i] = 1
            else:
                heappush(minheap, (nums[i], i))
            if wcount > k: # if added to maxheap it may have exceeded k, so need to remove from it
                _, idx = heappop(maxheap)
                if idx < l: continue 
                used[idx] = 0
                wsum -= nums[idx]
                wcount -= 1 
                heappush(minheap, (nums[idx], idx))
            if l >= 0: ans = min(ans, wsum)     
        return ans + start
```

# Leetcode BiWeekly Contest 123

## Maximum Good Subarray Sum

### Solution 1:  prefix sum, kadane's algorithm, last occurrence

```py
import math
class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        psum = list(accumulate(nums))
        vsum = Counter()
        last = {}
        ans = -math.inf
        for i in range(n):
            if nums[i] in last:
                l, r = last[nums[i]], i - 1
                vsum[nums[i]] += psum[r]
                if l > 0:
                    vsum[nums[i]] -= psum[l - 1]
                vsum[nums[i]] = max(0, vsum[nums[i]])
            for cand in [nums[i] - k, nums[i] + k]:
                if cand in last:
                    l, r = last[cand], i
                    segsum = vsum[cand] + psum[r]
                    if l > 0: segsum -= psum[l - 1]
                    ans = max(ans, segsum)
            last[nums[i]] = i
        return ans if ans > -math.inf else 0
```

## Find the Number of Ways to Place People II

### Solution 1:  sort, fenwick tree

```cpp
long long neutral = 0;
struct FenwickTree {
    vector<long long> nodes;
    
    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, long long val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return query(right) - query(left - 1);
    }

    long long query(int idx) {
        long long result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};
class Solution {
public:
    int numberOfPairs(vector<vector<int>>& points) {
        int n = points.size();
        sort(points.begin(), points.end(), [](const vector<int>& a, const vector<int>& b) {
                  if (a[0] == b[0]) // If the x-coordinates are equal
                      return a[1] > b[1]; // Sort by y-coordinate in descending order
                  return a[0] < b[0]; // Otherwise, sort by x-coordinate in ascending order
              });
        int ans = 0;
        unordered_map<int, int> y_coords;
        int idx = 1;
        vector<int> y_values;
        for (int i = 0; i < n; i++) {
            y_values.push_back(points[i][1]);
        }
        sort(y_values.begin(), y_values.end());
        for (int y : y_values) {
            if (y_coords.find(y) != y_coords.end()) continue;
            y_coords[y] = idx++;
        }
        FenwickTree fenwick;
        fenwick.init(idx);
        vector<int> between;
        for (int i = 0; i < n; i++) {
            int x1 = points[i][0], y1 = points[i][1];
            between.clear();
            for (int j = i + 1; j < n; j++) {
                int x2 = points[j][0], y2 = points[j][1];
                if (x1 <= x2 && y1 >= y2) {
                    int l = y_coords[y2], r = y_coords[y1];
                    if (fenwick.query(l, r) == 0) ans++;
                    between.push_back(y2);
                    fenwick.update(y_coords[y2], 1);
                }
            }
            for (int y : between) {
                fenwick.update(y_coords[y], -1);
            }
        }
        return ans;
    }
};
```

```py
class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def query_range(self, i, j):
        return self.query(j) - self.query(i - 1)

    def __repr__(self):
        return f"array: {self.sums}"
    
class Solution:
    def numberOfPairs(self, points: List[List[int]]) -> int:
        n = len(points)
        # x1 <= x2 and y1 >= y2
        points.sort(key = lambda point: (point[0], -point[1]))
        ans = 0
        # get index for y value
        y_coordinates = {}
        for _, y in sorted(points, key = lambda p: p[1]):
            if y in y_coordinates: continue
            y_coordinates[y] = len(y_coordinates) + 1
        m = len(y_coordinates)
        fenwick = FenwickTree(m)
        # x increasing, y decreasing
        for i in range(n):
            x1, y1 = points[i]
            between = []
            for j in range(i + 1, n):
                x2, y2 = points[j]
                if x1 <= x2 and y1 >= y2:
                    r, l = y_coordinates[y1], y_coordinates[y2]
                    if fenwick.query_range(l, r) == 0: 
                        ans += 1
                    between.append(y2)
                    y_index = y_coordinates[y2]
                    fenwick.update(y_index, 1)
            for y in between:
                y_index = y_coordinates[y]
                fenwick.update(y_index, -1)
        return ans
```

### Solution 2:  sort, binary search, sortedlist data structure

```py
from sortedcontainers import SortedList 
class Solution:
    def numberOfPairs(self, points: List[List[int]]) -> int:
        n = len(points)
        # x1 <= x2 and y1 >= y2
        points.sort(key = lambda point: (point[0], -point[1]))
        ans = 0
        # x increasing, y decreasing
        for i in range(n):
            x1, y1 = points[i]
            between = SortedList()
            for j in range(i + 1, n):
                x2, y2 = points[j]
                if x1 <= x2 and y1 >= y2:
                    idx = between.bisect_left(y2)
                    if idx == len(between): ans += 1
                    between.add(y2)
        return ans
```

### Solution 3:  sort, binary search, set

```cpp
class Solution {
public:
    int numberOfPairs(vector<vector<int>>& points) {
        int n = points.size();
        sort(points.begin(), points.end(), [](const vector<int>& a, const vector<int>& b) {
            if (a[0] == b[0]) return a[1] > b[1]; // sort y coordinate in descending order
            return a[0] < b[0]; // sort x coordinate in ascending order
        });
        int ans = 0;
        set<int> between;
        for (int i = 0; i < n; i++) {
            int x1 = points[i][0], y1 = points[i][1];
            between.clear();
            for (int j = i + 1; j < n; j++) {
                int x2 = points[j][0], y2 = points[j][1];
                if (x1 <= x2 && y1 >= y2) {
                    auto it = between.lower_bound(y2);
                    if (it == between.end()) ans++;
                    between.insert(y2);
                }
            }
        }
        return ans;
    }
};
```

### Solution 4:  sort, track max y value between two pairs, O(n^2)

```py
class Solution:
    def numberOfPairs(self, points: List[List[int]]) -> int:
        n = len(points)
        # x1 <= x2 and y1 >= y2
        points.sort(key = lambda point: (point[0], -point[1]))
        ans = 0
        # x weakly increasing, y weakly decreasing
        for i in range(n):
            x1, y1 = points[i]
            max_y = -math.inf
            for j in range(i + 1, n):
                x2, y2 = points[j]
                if x1 <= x2 and y1 >= y2: # max_y <= y1
                    if max_y < y2: ans += 1
                    max_y = max(max_y, y2)
        return ans
```



# Leetcode BiWeekly Contest 124

## Apply Operations to Make String Empty

### Solution 1:  counter, reverse

You just need to take the characters from s that achieve the maximum frequency and in the reverse order. 

```py
class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        ans = []
        counts = Counter(s)
        mx = max(counts.values())
        for ch in reversed(s):
            if counts[ch] == mx: 
                counts[ch] = 0
                ans.append(ch)
        return "".join(reversed(ans))
```

## Maximum Number of Operations With the Same Score II

### Solution 1:  dynamic programming

You can use interval dynamic programming to solve this problem.  Want to think about an iterative approach post contest.

```py
class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        @cache
        def dp(i, j, target):
            if j - i + 1 < 2: return 0
            ans = 0
            if i + 1 < n and nums[i] + nums[i + 1] == target:
                ans = max(ans, dp(i + 2, j, target) + 1)
            if j - 1 >= 0 and nums[j - 1] + nums[j] == target:
                ans = max(ans, dp(i, j - 2, target) + 1)
            if nums[i] + nums[j] == target:
                ans = max(ans, dp(i + 1, j - 1, target) + 1)
            return ans
        res = max(dp(2, n - 1, nums[0] + nums[1]), dp(1, n - 2, nums[0] + nums[-1]), dp(0, n - 3, nums[-1] + nums[-2]))
        return res + 1
```

## Maximize Consecutive Elements in an Array After Modification

### Solution 1:  sort, greedy

You are allowed to increment each integer by at most 1. 

Given an array [1,2,3,4,5]  It is easy right to calculate the longest consecutive subarray
What about array [1,2,3,4,7,8] It is obvious that 7 can never connect to 4, because the difference is greater than 2.
what about array [1,2,3,4,6,7] You have a different of 2 between 4 and 6.  And you have created 1,2,3,4 which is length of 4, you can increment them all by 1, then it will match up with 6, 7
What about array [1,1,3,4,6,7] Between the 1 and 3, it realizes it can take 3 right, cause 1,2,3
Then you get 1,2,3,4.  But 2,3,4,5,6 is not possible.  The reason is that you increment the 1 at the beginning. 
What about array [1,1,1,3,4,6,7] This one added in an extra 1 that is going to cause problems. cause it will be smaller than nxt by 2,  so just skip it. 



Observation 1:
Define a contiguous block to be a subarray where the difference between adjacent elements is less than 2.

Observation 2:
You can never connect to adjacent blocks if the difference between last element of left block and first element of right block is greater than 2. 

Observation 3:
You need to track the length of the longest normal consecutive elements, that is without any operation.  Because if you split between two blocks where difference = 2. You can take the longest normal sequence and add that to the start of this contiguous blocks. The reason is you can just increment every single value in the normal sequence to get it to start this new block.  So you can glue together previous to this block.

Observation 4:
[1,1,1] = [1,2]
You can ignore more than 2 of any elements.  It will be useless, it can be built in if you just skip any element that is less than what you are looking for even with an operation so if it is 2 less you can skip.


Observation x:
There are a few edge cases that will be difficult to handle
[1,1,3,4,4,6], but in this example you can see that you can chain together blocks b1, b2, b3.
[1,2,4,5,6,8,9] In this example you can chain together b1, b2 or b2, b3, but not all of them.
Let's change the definition of a contiguous block however.  
Preprocess the array above you'd get [1,2,3,4,5,6] so it is actually all a single block.  I think you should increment elements that are equal to their previous element.  

```py
class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = cur = nxt = norm = 0
        for i in range(n):
            if nums[i] + 1 < nxt: continue
            if nums[i] + 1 == nxt or nums[i] == nxt: #extend
                cur += 1
                nxt += 1
            elif i > 0 and nums[i] - nums[i - 1] == 2:
                cur = norm + 1
                nxt = nums[i] + 1
            else:
                cur = 1
                nxt = nums[i] + 1
            if i > 0 and nums[i] == nums[i - 1] + 1: norm += 1
            else: norm = 1
            ans = max(ans, cur)
        return ans
```



# Leetcode BiWeekly Contest 125

## 3066. Minimum Operations to Exceed Threshold Value II

### Solution 1:  min heap

```py
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        heapify(minheap := nums)
        ans = 0
        while minheap[0] < k:
            x, y = heappop(minheap), heappop(minheap)
            heappush(minheap, 2 * min(x, y) + max(x, y))
            ans += 1
        return ans
```

## 3067. Count Pairs of Connectable Servers in a Weighted Tree Network

### Solution 1:  combinatorics, dfs, tree

```py
class Solution:
    def countPairsOfConnectableServers(self, edges: List[List[int]], m: int) -> List[int]:
        n = len(edges) + 1
        ans = [0] * n
        adj = [[] for _ in range(n)]
        def dfs(u, p):
            sz[u] = 1 if dist[u] % m == 0 else 0
            for v, w in adj[u]:
                if v == p: continue
                dist[v] = dist[u] + w
                dfs(v, u)
                sz[u] += sz[v]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        for r in range(n):
            dist, sz = [0] * n, [0] * n
            dfs(r, -1)
            cnt = 0
            for v, _ in adj[r]:
                ans[r] += cnt * sz[v]
                cnt += sz[v]
        return ans
```

## 3068. Find the Maximum Sum of Node Values

### Solution 1:  tree, cancelation property of applying xor even, bit manipulation, parity

```py
class Solution:
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        ans = sum(max(x, x ^ k) for x in nums)
        cnt = sum(1 for x in nums if x ^ k > x)
        if cnt & 1: # remove smallest
            delta = min(max(x, x ^ k) - min(x, x ^ k) for x in nums)
            ans -= delta
        return ans
```

