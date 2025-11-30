# Leetcode Weekly contest 270

## 2094. Finding 3-Digit Even Numbers

### Solution: simplify 3 for loops by using fact only 3 digits and count the digits. 

```c++
vector<int> findEvenNumbers(vector<int>& digits) {
    int count[10] = {};
    for (int& dig : digits) {
        count[dig]++;
    }
    vector<int> arr;
    for (int i = 1;i<10;i++) {
        for (int j = 0;j<10 && count[i]>0;j++) {
            for (int k = 0;k<10 && count[j]>(i==j);k+=2) {
                if (count[k]>(j==k)+(i==k)) {
                    arr.push_back(i*100+j*10+k);
                }
            }
        }
    }
    return arr;
}
```

## 2095. Delete the Middle Node of a Linked List

### Solution: 

```c++
ListNode* deleteMiddle(ListNode* head) {
    if (!head->next) {
        return nullptr;
    }
    ListNode *slow = head, *fast = head, *prev = head;
    while (fast && fast->next) {
        prev = slow;
        slow=slow->next;
        fast=fast->next->next;
    }
    prev->next=slow->next;
    return head;
}
```

## 2096. Step-By-Step Directions From a Binary Tree Node to Another

### Solution: Find the LCA with DFS, reconstruct the paths with O(n) space

```c++
vector<vector<TreeNode*>> paths;
int start, end;
void dfs(TreeNode* root, vector<TreeNode*>& path) {
    if (!root) {
        return;
    }
    path.push_back(root);
    if (root->val==start) {
        paths[0]=path;
    }
    if (root->val==end) {
        paths[1]=path;
    }
    dfs(root->left, path);
    dfs(root->right, path);
    path.pop_back();
}
string getDirections(TreeNode* root, int startValue, int destValue) {
    start = startValue, end = destValue;
    vector<TreeNode*> path;
    paths.resize(2);
    dfs(root, path);
    TreeNode* lca;
    int ups = 0;
    for (int i = 0;;i++) {
        if (i==paths[1].size()) {
            ups = paths[0].size()-i;
            lca = paths[1].back();
            break;
        }
        if (i==paths[0].size()) {
            lca = paths[0].back();
            break;
        }
        if (paths[0][i]!=paths[1][i]) {
            ups = paths[0].size()-i;
            lca = paths[1][i-1];
            break;
        }
    }
    string directions;
    while (ups--) {
        directions += 'U';
    }
    for (int i = 0;i<paths[1].size();i++) {
        if (lca->left==paths[1][i]) {
            directions += 'L';
            lca = lca->left;
        } else if (lca->right==paths[1][i]) {
            directions += 'R';
            lca=lca->right;
        }
    }
    return directions;
}
```

### Solution: LCA with O(1) space? can that be used here? 



## 2097. Valid Arrangement of Pairs

### Solution: Eulerian path/circuit with Hierzholzer's algorithm, indegrees,outdegrees + postorder dfs and reverse traversed edges.
Eulerian path is when you visit each edge exactly once.

```c++
vector<vector<int>> path;
unordered_map<int,int> indegrees,outdegrees;
unordered_map<int,vector<int>> graph;
void dfs(int node) {
    while (outdegrees[node]) {
        outdegrees[node]--;
        int nei = graph[node][outdegrees[node]];
        dfs(nei);
        path.push_back({node,nei});
    }
}
vector<vector<int>> validArrangement(vector<vector<int>>& pairs) {
    unordered_set<int> nodes;
    for (auto& pir : pairs) {
        indegrees[pir[1]]++;
        outdegrees[pir[0]]++;
        nodes.insert(pir[0]);
        nodes.insert(pir[1]);
        graph[pir[0]].push_back(pir[1]); // directed edge form 0 -> 1 
    }
    int start = pairs[0][0];
    for (auto& node:nodes) {
        if (outdegrees[node]-indegrees[node]==1) {
            start = node;
            break;
        }
    }
    dfs(start);
    reverse(path.begin(),path.end());
    return path;
}
```

# Leetcode Weekly contest 272

## LC2108. Find First Palindromic String in the Array

### Solution:  Reverse string

Check if it is palindrome by reversing each word and return the first one that is palindrome


```c++
bool isPalindrome(string& word) {
    string rword = "";
    int n = word.size();
    for (int i = n-1;i>=0;i--) {
        rword += word[i];
    }
    return word==rword;
}
string firstPalindrome(vector<string>& words) {
    for (string &word : words) {
        if (isPalindrome(word)) {
            return word;
        }
    }
    return "";
}
```

```c++
string firstPalindrome(vector<string>& words) {
    for (string &word : words) {
        if (word==string(word.rbegin(),word.rend())) {
            return word;
        }
    }
    return "";
}
```

```py
def firstPalindrome(self, words: List[str]) -> str:
    for w in words:
        if w==w[::-1]:
            return w   
    return ""
```

### Solution: Two pointers to find palindromes

Efficient solution, avoid creating temporary strings.  

```c++
bool isPalindrome(string& s) {
    int n = s.size();
    for (int i = 0,j = n-1;i<j;i++,j--) {
        if (s[i]!=s[j]) {return false;}
    }
    return true;
}
string firstPalindrome(vector<string>& words) {
    for (string &word : words) {
        if (isPalindrome(word)) {
            return word;
        }
    }
    return "";
}
```

## LC2109. Adding Spaces to a String

### Solution: String 

I am able to use the reserve to set the capacity to reduce the time for reallocating the string when it reaches capacity. 
This can be done because I know the final size of the string. 

```c++
string addSpaces(string s, vector<int>& spaces) {
    string ret;
    int n = s.size(), m = spaces.size();
    ret.reserve(n+m);
    for (int i = 0,j=0;i<n;i++) {
        if (j<spaces.size() && spaces[j]==i) {
            ret += " ";
            j++;
        }
        ret += s[i];
    }
    return ret;
}
```

In python it is a little trickier with strings because it recreates a string when you add to it,
so you have to use an array instead and append to it, then reconstruct the string from the array. 

```py
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        store = []
        store.append(s[:spaces[0]])
        for i in range(1,len(spaces)):
            store.append(' ')
            store.append(s[spaces[i-1]:spaces[i]])
        store.append(' ')
        store.append(s[spaces[-1]:])
        return "".join(store)
```

## LC2110. Number of Smooth Descent Periods of a Stock


### Solution: Iterative

```c++
long long getDescentPeriods(vector<int>& prices) {
    long long num = 1, len = 1;
    int n= prices.size();
    for (int i = 1;i<n;i++) {
        len = prices[i-1]-prices[i]==1 ? len+1 : 1;
        num+=len;
    }
    return num;
}
```

## LC2111. Minimum Operations to Make the Array K-Increasing

This problem you can use the idea of longest non-decreasing subsequence to solve it. 

This requires the O(nlogn) solution to that type of problem with the patience algorithm

I split the array up into k arrays that I will then compute the length of the longest non-decreasing subsequence in each array and subtract
it from the length of each array. 

take 
[12,6,12,6,14,2,13,17,3,8,11,7,4,11,18,8,8,3] with k=1
We find the length of the longest non-decreasing subsequence such as 2,3,8,11,11,18, which is of size 6
Now all we have to do is change the rest of the values to make the entire subarray non-decreasing, so yeah we only need to 
change 12.  

I guess since you are finding the minimum operations to make the array non-decreasing, you find the longest non-decreasing subsequence
and just change the rest of the values.  



### Solution: binary search with vector to find the k longest non-decreasing subsequences

```c++
const int NEUTRAL = 1e9;
int patience(vector<int>& arr) {
    int n = arr.size(), len = 0;
    vector<int> T(n,NEUTRAL);
    for (int &num : arr) {
        int i = upper_bound(T.begin(),T.end(), num) - T.begin();
        len = max(len, i+1);
        T[i] = num;
    }
    return len;
}
int kIncreasing(vector<int>& arr, int k) {
    vector<int> karray;
    int n = arr.size(), cnt = 0;
    for (int i = 0;i<k;i++) {
        karray.clear();
        for (int j = i;j<n;j+=k) {
            karray.push_back(arr[j]);
        }
        cnt += (karray.size()-patience(karray));
    }
    return cnt;
}
```

### Solution: This one is same but using a monostack 


```c++
int kIncreasing(vector<int>& arr, int k) {
    vector<int> karray;
    int n = arr.size(), longest = 0;
    for (int i = 0;i<k;i++) {
        vector<int> mono;
        for (int j = i;j<n;j+=k) {
            if (mono.empty() || mono.back()<=arr[j]) {
                mono.push_back(arr[j]);
            } else {
                *upper_bound(mono.begin(),mono.end(),arr[j]) = arr[j];
            }
        }
        longest += mono.size();
    }
    return arr.size()-longest;
}
```


I need practice using bisect_right in python, this is equivalent to upper_bound that is it finds the index of the first element
that is strictly greater than what you are searching for.  

```py
def kIncreasing(self, arr: List[int], k: int) -> int:
    def LNDS(arr):
        mono = []
        for x in arr:
            if not mono or mono[-1]<=x:
                mono.append(x)
            else:
                mono[bisect_right(mono, x)] = x
        return len(mono)
    return len(arr) - sum(LNDS(arr[i::k]) for i in range(k))
```

# Leetcode Weekly Contest 273

## 2119. A Number After a Double Reversal

### Solution 1: check that number doesn't have trailing zeros via it is not divisible by 10

```c++
bool isSameAfterReversals(int num) {
    return num==0 || num%10!=0;
}
```

## 2120. Execution of All Suffix Instructions Staying in a Grid

### Solution 1: Brute force algorithm to execute all instructions until out of bounds

```c++
vector<int> executeInstructions(int n, vector<int>& startPos, string s) {
    int m = s.size();
    vector<int> results(m,0);
    auto inBounds = [&](const auto i, const auto j) {
        return i>=0 && i<n && j>=0 && j<n;  
    };
    for (int i = 0;i<m;i++) {
        int steps = 0;
        for (int j = i, row = startPos[0], col = startPos[1];j<m;j++) {
            col += (s[j]=='R');
            col -= (s[j]=='L');
            row += (s[j]=='D');
            row -= (s[j]=='U');
            if (!inBounds(row,col)) {
                break;
            }
            steps++;
        }
        results[i] = steps;
    }
    return results;
}
```

### Solution 2:  O(m) with backtracking?

```c++

```


## 2121. Intervals Between Identical Elements

### Solution 1: Math, prefix ans suffix 

```c++
vector<long long> getDistances(vector<int>& arr) {
    unordered_map<int,vector<long long>> indicesMap;
    vector<long long> results((int)arr.size(),0);
    for (int i = 0;i<arr.size();i++) {
        indicesMap[arr[i]].push_back(i);
    }
    for (auto &[_, idx]: indicesMap) {
        long long prefix = accumulate(idx.begin(),idx.end(),0LL), n = idx.size();
        long long suffix = n*idx[0];
        results[idx[0]] = total - negate;
        for (int i = 1,j=n-1;i<n;i++,j--) {
            prefix += (idx[i]-idx[i-1])*i;
            suffix -= (j*idx[i-1]-j*idx[i]);
            results[idx[i]] = prefix - suffix;
        }
    }
    return results;
}
```

## LC 2122. Recover the Original Array

### Solution 1: brute force through k with hash map for frequency and build result

```c++
vector<int> recoverArray(vector<int>& nums) {
    unordered_map<int,int> freqs;
    for (int &num : nums) {
        freqs[num]++;
    }
    int n = nums.size();
    sort(nums.begin(),nums.end());
    int start = nums[0];
    for (int k = 1; ; ++k) {
        auto it = lower_bound(nums.begin(),nums.end(),start+2*k);
        int diff = *it-start;
        if (diff%2==1) {
            k = diff/2;
            continue;
        }
        k = diff/ 2;
        vector<int> res;
        auto ff = freqs;
        for (int &num: nums) {
            if (ff[num]==0) continue;
            ff[num]--;
            if (ff[num+diff]--==0) {
                break;
            }
            res.push_back(num+k);
        }
        if (res.size()==n/2)
            return res;
    }
    return {};add and remove 
}
```
### Solution 2: Instead of looping through k, we loop through all all elements and pair up with the 0th 


```c++
vector<int> recoverArray(vector<int>& nums) {
    unordered_map<int,int> freqs;
    for (int &num : nums) {
        freqs[num]++;
    }
    int n = nums.size();
    sort(nums.begin(),nums.end());
    int start = nums[0];
    for (int i = 1, k;i<n;i++) {
        int diff = nums[i]-nums[0];
        if (diff==0 || diff%2==1) {
            continue;
        }
        k = diff/ 2;
        vector<int> res;
        auto ff = freqs;
        for (int &num: nums) {
            if (ff[num]==0) continue;
            ff[num]--;
            if (ff[num+diff]--==0) {
                break;
            }
            res.push_back(num+k);
        }
        if (res.size()==n/2)
            return res;
    }
    return {};
}
```


### Solution 3: Using multiset with similar idea as above

```c++
vector<int> recoverArray(vector<int>& nums) {
    multiset<int> s(nums.begin(),nums.end());
    int start = *s.begin();
    for (int k = 1;;k++) {
        auto it = s.lower_bound(start+2*k);
        k = (*it-start)/2;
        if (start+2*k!=*it) continue;
        vector<int> recovered;
        auto ss = s;
        while (!ss.empty()) {
            auto it_lower = ss.begin();
            auto it_higher = ss.find(*it_lower+2*k);
            if (it_higher==ss.end()) break;
            recovered.push_back(*it_lower+k);
            ss.erase(it_lower);
            ss.erase(it_higher);
        }
        if (ss.empty()) {
            return recovered;
        }
    }
    return {};
}
```

# Leetcode Weekly Contest 274

## 2124. Check if All A's Appears Before All B's

### Solution: Retrun false if you find "ba" in the string

```c++
bool checkString(string s) {
    return s.find("ba")==string::npos;
}
```

```py
def checkString(self, s: str) -> bool:
    return s.find("ba")==-1
```

## 2125. Number of Laser Beams in a Bank

### Solution: count 1s in each row

```py
class Solution:
    def numberOfBeams(self, bank: List[str]) -> int:
        counts = map(lambda row: row.count("1"), bank)
        counts = list(filter(lambda row: row > 0, counts))
        return sum(counts[i] * counts[i - 1] for i in range(1, len(counts)))
```

## 2126. Destroying Asteroids

### Solution: sort asteroids and greedily destroy them until it is greater than total mass so far

```c++
const int INF = 1e5;
bool asteroidsDestroyed(int mass, vector<int>& asteroids) {
    int n = asteroids.size();
    sort(asteroids.begin(),asteroids.end());
    for (int asteroid : asteroids) {
        if (asteroid>mass) {return false;}
        if (mass>=INF) {return true;}
        mass+=asteroid;
    }
    return true;
}
```

## 2127. Maximum Employees to Be Invited to a Meeting

### Solution: Union Find (disjoint set) + topological sort with bfs

There are two cases:
1) The answer is the longest cycle
2) The answer is the sum of the longest acyclic path on all connected components with a 2-cycle. 
For this problem we put all of the nodes that are in a cycle in a disjoint set, this allows us to easily find the largest cycle in the functional
successor graph.  
We used a bfs for topological sort to find the length of acyclic paths, that I store in dist[i] the longest length traversed to node i.  
This way when we find 2-cycles we add them all to the sum for the dist[i]+dist[favorite[i]].

```c++
struct UnionFind {
    vector<int> parents, size;
    void init(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    bool uunion(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            return false;
        }
        return true;
    }
};
class Solution {
public:
    int maximumInvitations(vector<int>& favorite) {
        int n = favorite.size();
        vector<int> indegrees(n,0), dist(n,1);
        for (int i = 0;i<n;i++) {
            indegrees[favorite[i]]++;
        }
        UnionFind ds;
        ds.init(n);
        queue<int> q;
        for (int i = 0;i<n;i++) {
            if (!indegrees[i]) {
                q.push(i);
            }
        }
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            int v = favorite[node];
            dist[v] = max(dist[v], dist[node]+1);
            if (--indegrees[v]==0) {
                q.push(v);
            }
        }
        for (int i = 0;i<n;i++) {
            if (indegrees[i]) {
                ds.uunion(i,favorite[i]);
            }
        }
        int sum = 0, maxCycle = 0;
        for (int i = 0;i<n;i++) {
            if (!indegrees[i]) continue; // only want to consider those in cycle
            int len = ds.size[ds.find(i)];
            if (len==2) {
                indegrees[favorite[i]]--; // so doesn't start from this cycle as well. avoid double counting
                sum += dist[i]+dist[favorite[i]];
            } else {
                maxCycle = max(maxCycle, ds.size[ds.find(i)]);
            }
        }
        return max(maxCycle, sum);
    }
};
```

### Solution: DFS 
DFS for finding the longest cycle
DFS for finding the longest acyclic path attached to the 2-cycles.  

```c++

```

# Leetcode Weekly Contest 275

## 2133. Check if Every Row and Column Contains All Numbers

### Solution: array - rows and columns count O(n^2) and O(n) memory

```c++
bool checkValid(vector<vector<int>>& matrix) {
    int n = matrix.size();
    int rows[n+1], cols[n+1];
    for (int i = 0;i<n;i++) {
        memset(rows,0,sizeof(rows));
        memset(cols,0,sizeof(cols));
        for (int j = 0;j<n;j++) {
            if (++rows[matrix[j][i]]>1) {return false;}
            if (++cols[matrix[i][j]]>1) {return false;}
        }
    }
    return true;
}
```

## 2134. Minimum Swaps to Group All 1's Together II

### Solution: sliding window algorithm count number of 0s in the window. 
Every 0 must be replaced with a 1. 

```c++
int minSwaps(vector<int>& nums) {
    int n = nums.size();
    int ones = count(nums.begin(),nums.end(),1), best = n;
    for (int i=0,j=0, cnt = 0;i<n;i++) {
        while (j-i<ones) {
            cnt += nums[j++%n];
        }
        best = min(best, ones-cnt);
        cnt -= nums[i];           
    }
    return best;
}
```

## 2135. Count Words Obtained After Adding a Letter

An observation that made this easier to approach is that you store all the strings from startWords in some datastructure. 
Then when you iterate through the targetWords, remove one character from it and check it's existence in chosen datastructures. 


### Solution: Hashset + Bitmask
O(n+m) linear solution because number of characters = 26 and is constant. 

```c++
int wordCount(vector<string>& startWords, vector<string>& targetWords) {
    unordered_set<int> masks;
    int cnt = 0, mask;
    for (string& s : startWords) {
        mask = 0;
        for (char& c : s) {
            mask += (1<<(c-'a'));
        }
        masks.insert(mask);
    }
    for (string& s : targetWords) {
        for (int i = 0;i<s.size();i++) {
            mask = 0;
            for (int j = 0;j<s.size();j++) {
                if (i==j) continue;
                mask += (1<<(s[j]-'a'));
            }
            if (masks.count(mask)) {
                cnt++;
                break;
            }
        }
    }
    return cnt;
}
```

### Solution: Hashset + bitmask with bitsets
I feel using bitsets can scale for if you have more than 26 characters, without the bit limitation.

```c++
int wordCount(vector<string>& startWords, vector<string>& targetWords) {
    unordered_set<bitset<26>> masks;
    int cnt = 0;
    bitset<26> mask;
    for (string& s : startWords) {
        mask.reset();
        for (char& c : s) {
            mask.set(c-'a');
        }
        masks.insert(mask);
    }
    for (string& s : targetWords) {
        for (int i = 0;i<s.size();i++) {
            mask.reset();
            for (int j = 0;j<s.size();j++) {
                if (i==j) continue;
                mask.set(s[j]-'a');
            }
            if (masks.count(mask)) {
                cnt++;
                break;
            }
        }
    }
    return cnt;
}
```

### Solution:  Hashset + bitmask with bitsets
O(len(startWords)*len(startWords[0])*26)

```c++
int wordCount(vector<string>& startWords, vector<string>& targetWords) {
    unordered_set<bitset<26>> appears;
    int cnt = 0;
    for (string& s : startWords) {
        bitset<26> bs;
        for (char c = 'a';c<='z';c++) {
            if (s.find(c)!=string::npos) {
                bs.set(c-'a');
            }
        }
        for (char c = 'a';c<='z';c++) {
            if (!bs.test(c-'a')) {
                bitset<26> b = bs;
                b.set(c-'a');
                appears.insert(b);
            }
        }
    }
    for (string& s : targetWords) {
        bitset<26> bc;
        for (char c='a';c<='z';c++) {
            if (s.find(c)!=string::npos) {
                bc.set(c-'a');
            }
        }
        cnt += appears.count(bc);
    }
    return cnt;
}
```

### Solution: Trie Datastructure

```c++
struct Node {
    int children[26];
    bool isLeaf;
    void init() {
        memset(children,0,sizeof(children));
        isLeaf = false;
    }
};
struct Trie {
    vector<Node> trie;
    void init() {
        Node root;
        root.init();
        trie.push_back(root);
    }
    void insert(string& s) {
        int cur = 0;
        for (char &c : s) {
            int i = c-'a';
            if (trie[cur].children[i]==0) {
                Node root;
                root.init();
                trie[cur].children[i] = trie.size();
                trie.push_back(root);
            }
            cur = trie[cur].children[i];
        }
        trie[cur].isLeaf= true;
    }
    int search(string& s) {
        int cur = 0;
        for (char &c : s) {
            int i = c-'a';
            if (!trie[cur].children[i]) { return false;
            }
            cur = trie[cur].children[i];
        }
        return trie[cur].isLeaf;
    }
};
class Solution {
public:
    int wordCount(vector<string>& startWords, vector<string>& targetWords) {
        Trie trie;
        trie.init();
        int cnt = 0;
        unordered_set<string> seen;
        for (string& s : startWords) {
            sort(s.begin(),s.end());
            trie.insert(s);
        }
        for (string& s : targetWords) {
            sort(s.begin(),s.end());
            for (int i = 0;i<s.size();i++) {
                string tmp = s.substr(0,i) + s.substr(i+1); // remove the ith character, and check it exists
                if (trie.search(tmp)) {
                    cnt++;
                    break;
                }
            }
        }
        return cnt;
    }
};
```

### Solution: Hashset

```c++
int wordCount(vector<string>& startWords, vector<string>& targetWords) {
    unordered_set<string> seen;
    int cnt = 0;
    for (string& s : startWords) {
        sort(s.begin(),s.end());
        seen.insert(s);
    }
    for (string& s : targetWords) {
        sort(s.begin(),s.end());
        for (int i = 0;i<s.size();i++) {
            if (seen.count(s.substr(0,i)+s.substr(i+1))) {
                cnt++;
                break;
            }
        }
    }
    return cnt;
}
```

## 2136. Earliest Possible Day of Full Bloom

### Solution: sorting - greedy plant the flower seeds that have the longest growing time first. 


```c++
int earliestFullBloom(vector<int>& P, vector<int>& G) {
    int n = P.size();
    vector<int> plants(n);
    iota(plants.begin(), plants.end(),0);
    sort(plants.begin(),plants.end(),[&](const auto& i, const auto& j) {
        return G[i]>G[j];
    });
    int finish = 0;
    for (int i = 0, time = 0;i<n;i++) {
        int index = plants[i];
        time += P[index];
        finish = max(finish, time+G[index]);
    }
    return finish;
}
```

### Solution 1: greedy + sort

```py
class Solution:
    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        bloom_time = time = 0
        arr = sorted([(x,y) for x,y in zip(plantTime, growTime)], key = lambda x: (-x[1],x[0]))
        for pl, gr in arr:
            time += pl
            bloom_time = max(bloom_time, time + gr)
        return bloom_time
```

### Solution: multiset with custom comparator for plant sort descending order for growing and if tied, ascending order for planting.


```c++
struct Plant {
    int grow,seed;
    void init(int g, int s) {
        grow=g;
        seed=s;
    }
    bool operator<(const Plant& b) const {
        return grow>b.grow;
    }
};
class Solution {
public:
    int earliestFullBloom(vector<int>& P, vector<int>& G) {
        int n = P.size();
        multiset<Plant> plantSet;
        for (int i = 0;i<n;i++) {
            Plant pl;
            pl.init(G[i],P[i]);
            plantSet.insert(pl);
        }
        int finish = 0, time = 0;
        for (auto& plant : plantSet) {
            time += plant.seed;
            finish = max(finish, time+plant.grow);
        }
        return finish;
    }
};
```

# Leetcode Weekly Contest 276


## 2138. Divide a String Into Groups of Size k

### Solution: Array

```c++
vector<string> divideString(string s, int k, char fill) {
    int n = s.size();
    vector<string> arr;
    for (int i = 0;i<n;i+=k) {
        arr.push_back(s.substr(i,k));
    }
    for (int i = arr.back().size();i<k;i++) {
        arr.back()+=fill;
    }
    return arr;
}
```


## 2139. Minimum Moves to Reach Target Score

### Solution: backward thinking + greedy (choose divide by 2 first)

```c++
int minMoves(int target, int maxDoubles) {
    int moves = 0;
    while (target>1 && maxDoubles--) {
        moves += 1 + target%2;
        target/=2;
    }
    return moves + target-1;
}
```


## 2140. Solving Questions With Brainpower

### Solution: dynamic programming + memoization

```c++
long long recurse(int i, vector<vector<int>>& Q, vector<long long>& dp) {
    int n = Q.size();
    if (i>=n) {return 0;}
    if (dp[i]!=-1) {
        return dp[i];
    }
    return dp[i] = max(Q[i][0]+recurse(i+Q[i][1]+1,Q,dp), recurse(i+1,Q,dp));
}
long long mostPoints(vector<vector<int>>& questions) {
    vector<long long> points((int)questions.size(), -1);
    return recurse(0,questions,points);
}
```

```py
class Solution:
    def mostPoints(self, questions: List[List[int]]) -> int:
        n = len(questions)
        dp = [0]*(n + 1)
        for i in range(n - 1, -1, -1):
            pts, bp = questions[i]
            skip = dp[i + 1]
            take = dp[min(n, i + bp + 1)] + pts
            dp[i] = max(take, skip)
        return dp[0]
```

## 2141. Maximum Running Time of N Computers

### Solution: binary search

```c++

```

# Leetcode Weekly Contest 277

## 2148. Count Elements With Strictly Smaller and Greater Elements

### Solution: Compare all elements to min and max in array and check it is strictly between

TC: O(n)

```py
def countElements(self, nums: List[int]) -> int:
    mx, mn = max(nums), min(nums)
    return sum(1 for num in nums if num>mn and num<mx)
```

```c++
int countElements(vector<int>& nums) {
    int mn = INT_MAX, mx = INT_MIN;
    for (int& num : nums) {
        mn = min(mn, num);
        mx = max(mx,num);
    }
    return accumulate(nums.begin(),nums.end(),0,[&](const auto& a, const auto& b) {
        return a + (b>mn && b<mx);
    });
}
```

## 2149. Rearrange Array Elements by Sign

### Solution: Two pointers


```c++
vector<int> rearrangeArray(vector<int>& nums) {
    int n = nums.size();
    vector<int> pos
    for (int i = 0,pptr = 0, nptr = 1;i<n;i++) {
        if (nums[i]>0) {
            res[pptr] = nums[i];
            pptr+=2;
        } else {
            res[nptr] = nums[i];
            nptr+=2;
        }
    }
    return res;
}
```

## 2150. Find All Lonely Numbers in the Array

### Solution: hashmap 

```c++
vector<int> findLonely(vector<int>& nums) {
    unordered_map<int,int> count;
    for (int& num : nums) {
        count[num]++;
    }
    vector<int> res;
    for (int& num : nums) {
        if (count[num]==1 && count[num-1]+count[num+1]==0) {
            res.push_back(num);
        }
    }
    return res;
}
```

```py
def findLonely(self, nums: List[int]) -> List[int]:
    count = Counter(nums)
    return [num for num in nums if count[num]==1 and count[num-1]+count[num+1]==0]
```

## 2151. Maximum Good People Based on Statements

### Solution: bitmask + loop to check statements are valid

TC: O(n^2 * 2^n)

```c++
int maximumGood(vector<vector<int>>& statements) {
    int n = statements.size(), best = 0;
    auto valid = [&](int mask) {
        for (int i = 0;i<n;i++) {
            if ((mask>>i)&1) {
                for (int j = 0;j<n;j++) {
                    int good = (mask>>j)&1;
                    if ((statements[i][j]==0 && good) || (statements[i][j]==1 && !good)) return false;
                }
            }
        }  
        return true;
    };
    for (int i = 1; i<(1<<n);i++) {
        if (valid(i)) {
            best = max(best, __builtin_popcount(i));
        }
    }
    return best;
}
```

### Solution: dfs + backtracking (check if valid and update best answer with count of good)

```c++
int best;
bool valid(string& s, vector<vector<int>>& statements) {
    int n = statements.size();
    for (int i = 0;i<n;i++) {
        if (s[i]=='1') {
            for (int j = 0;j<n;j++) {
                int good = s[j]-'0';
                if ((statements[i][j]==0 && good) || (statements[i][j]==1 && !good)) return false;
            }
        }
    }  
    return true;
}
void dfs(vector<vector<int>>& statements, string& state, int i, int cnt) {
    if (i==statements.size()) {
        if (valid(state,statements)) {
            best = max(best, cnt);
        }
        return;
    }
    state += '0'; // assume bad person
    dfs(statements,state,i+1,cnt);
    state[i] = '1'; // assume good person
    dfs(statements,state,i+1,cnt+1);
    state.pop_back();
}
int maximumGood(vector<vector<int>>& statements) {
    string cur = "";
    best = 0;
    cur.reserve((int)statements.size());
    dfs(statements,cur,0,0);
    return best;
}
```

# Leetcode Weekly Contest 278


last problem is just bitmask with 26 characters
and also it is just bitset as well, it is easy. 

# Leetcode Weekly Contest 281

## 2180. Count Integers With Even Digit Sum

### Solution: Convert to string and sum the digits and check if it is even

```py
class Solution:
    def countEven(self, num: int) -> int:
        return sum(1 for x in range(2,num+1) if sum(map(int,str(x)))%2==0)
```

## 2181. Merge Nodes in Between Zeros

### Solution: Two pointer, slow and fast

```py
class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head.next
        sum_ = 0
        while fast:
            sum_ += fast.val
            if fast.val == 0:
                slow.next = ListNode(sum_)
                slow = slow.next
                sum_ = 0
            fast = fast.next
        return head.next
```

## 2182. Construct String With Repeat Limit

### Solution: max Heap datastructure with constraints applied 

```py
from heapq import heappop, heappush
class Solution:
    def repeatLimitedString(self, s: str, repeatLimit: int) -> str:
        Character = namedtuple('Character', ['char', 'count'])
        Character.__lt__ = lambda self, other: self.char > other.char
        freq = Counter(s)
        heap = [Character(ch, cnt) for ch, cnt in freq.items()]
        heapify(heap)
        res = []
        while len(heap) > 0:
            ch = heappop(heap)
            upper_bound = min(repeatLimit, ch.count)
            for _ in range(upper_bound):
                res.append(ch.char)
            if upper_bound == ch.count: continue
            if len(heap) == 0: break
            replacech = heappop(heap)
            res.append(replacech.char)
            if replacech.count > 1:
                heappush(heap, Character(replacech.char, replacech.count - 1))
            heappush(heap, Character(ch.char, ch.count - upper_bound))
        return "".join(res)
```

### Solution: Simply construct string from descending order and look at next whenever reach repeatLimit

```py

```

## 2183. Count Array Pairs Divisible by K

### Solution: Factorization + hashmap

Find the prime factorization for K, and every integer in nums. 
Use that to find the difference between both prime factorizations.
The difference will be the needed prime factors. We can create
a power set from that or every possible set of those prime factors, or 
all subsets, and get the product of each one and increase the counter by that

So then all we need is to use the counter and find how many times
we've seen this integer previously.

This solution is valid but it TLE, it needs to be optimized to pass on 
Leetcode

```py
from math import sqrt
from numpy import prod
class Solution:
    def prime_factors(self, num):
        res = []
        i = 2
        while num > 1: 
            while num%i==0:
                num//=i
                res.append(i)
            i += 1
        return res
    
    # COMPUTES THE FACTORS WE NEED TO MAKE CURRENT NUMBER NUMS2 DIVISIBLE BY K
    def difference(self, nums1, nums2):
        res = []
        i, j = 0, 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                res.append(nums1[i])
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                i += 1
                j += 1
        while i < len(nums1):
            res.append(nums1[i])
            i += 1
        return res
    
    def countPairs(self, nums: List[int], k: int) -> int:
        pk = self.prime_factors(k)
        counter = Counter()
        cnt = 0

        for num in nums:
            nk = self.prime_factors(num)
            needed_factors = self.difference(pk,nk)

            # COUNTER OF TIMES THIS PROD HAS BEEN SEEN
            cnt += counter[prod(needed_factors)]
            cset = []
            # POWER SET
            def subset(start):
                counter[prod(cset)] += 1
                if start == len(nk): return
                for i in range(start, len(nk)):
                    if i > start and nk[i] == nk[i-1]: continue
                    cset.append(nk[i])
                    subset(i+1)
                    cset.pop()
            
            # UPDATING COUNTER
            subset(0)
        return cnt
```

### Solution: GCD + Hashmap

Since we could have at most 128 factors for k possibly, we can say counter will have at most 128 elements
so it is N*N, where N = number of factors of k.  But that is really low, so it works

if
a*b%k==0 then
gcd(a,k)*gcd(b,k)%k==0 

GCD is O(log(min(a,b))) if it uses euclidean algorithm

so we have O(N*log(100000 + N*K),  where N = len(nums), K = all possible gcd

```py
from math import gcd
class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        counter = Counter(gcd(num, k) for num in nums)
        cnt = 0
        for a in counter:
            for b in counter:
                if a <= b and a*b%k == 0:
                    cnt += counter[a] * counter[b] if a < b else counter[a] * (counter[a] - 1) // 2
        return cnt
```



```py
from math import gcd
class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        counter = Counter()
        res = 0
        for num in nums:
            curgcd = gcd(num, k)
            for g, cnt in counter.items():
                if g*curgcd%k==0:
                    res += cnt
            counter[curgcd] += 1
        return res
```

# Leetcode Weekly Contest 282

## 2185. Counting Words With a Given Prefix

### Solution: loop

```py
class Solution:
    def prefixCount(self, words: List[str], pref: str) -> int:
        return sum(1 for word in words if word[:len(pref)] == pref)
```

## 2186. Minimum Number of Steps to Make Two Strings Anagram II

### Solution: hashmap + loop

```py
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        f1, f2 = [0]*26, [0]*26
        for c in s:
            f1[ord(c)-ord('a')] += 1
        for c in t: 
            f2[ord(c)-ord('a')] += 1
        return sum(abs(c1-c2) for c1, c2 in zip(f1, f2))
```

## 2187. Minimum Time to Complete Trips

### Solution 1: binary search 

```py
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        lo, hi = 1, min(time)*totalTrips
        def possible(est):
            return sum(est//t for t in time) >= totalTrips
            
        while lo < hi:
            mid = (lo+hi) >> 1
            if possible(mid):
                hi = mid
            else:
                lo = mid + 1
            
        return lo
```

### Solution 2: binary search + bisect + greedy

```py
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        return bisect.bisect_left(range(min(time)*totalTrips), totalTrips, key = lambda ctime: sum((ctime//t for t in time)))
```

## 2188. Minimum Time to Finish the Race

I should graph what is happening with a tire over time do a little bit of data analysis.

But first let's see if I can make some headway with this knowledge

So I know the time for remaining on a tire grows at an exponential rate

I can have 100,000 tires potentially to choose from 
I can perform a changeTime at any point, but we don't want to do that, it will be too 
complex of problem

One solution is to pack all the possiblities into a min_heap

But let's consider this as a graph

what if we have all our initial nodes for lap 1, which could be any of the 100,000 tires
Then we have to traverse the node to the next lap, now we either traverse with same tire and 
have an edge weight that is the cost of lap 2, or we change tires and have that cost. 

Anyway all of my assumptions were wrong, I should have realized that just map
the best price to continue to go straight for sometime, and realize the 
worst case is we'd want to go straight for 18 times,  via an analysis of upper bounds
if r=2, the cheapest, at the point of 2^18 > max(changeTime) + max(f) = 2e5
Even if it were the most expensive change time and slowest fresh tires, at some point, 
with the smallest r, it will no longer make since after 18 laps on any tire to continue. 
So we can assume we only go straight upto 18 times.  

### Solution: Brute force for going straight + dynamic programming for computing minimum cost for each lap

TC: O(N^2), where N = numLaps

```py
class Solution:
    def minimumFinishTime(self, tires: List[List[int]], changeTime: int, numLaps: int) -> int:
        nochange = [math.inf]*19
        LIMIT = int(2e5)
        for f, r in tires:
            cur_time = f
            total_time = cur_time
            nochange[1] = min(nochange[1], total_time)
            for i in range(2,19):
                cur_time *= r
                total_time += cur_time
                if total_time > LIMIT: break
                nochange[i] = min(nochange[i],total_time)
        
        dp = [math.inf]*(numLaps+1)
        for i in range(1,numLaps+1):
            if i<19:
                dp[i] = min(dp[i], nochange[i]) 
            for j in range(1,i//2+1):
                dp[i] = min(dp[i], dp[j] + changeTime + dp[i-j])
                
        return dp[-1]
```



# Leetcode Weekly Contest 283

## 2194. Cells in a Range on an Excel Sheet

### Solution: Iterate through strings

```py
class Solution:
    def cellsInRange(self, s: str) -> List[str]:
        start, end = s.split(':')
        res = []
        for col in range(ord(start[0]), ord(end[0])+1):
            for row in range(ord(start[1]), ord(end[1])+1):
                res.append(chr(col)+chr(row))
        return res
```

## 2195. Append K Integers With Minimal Sum

### Solution: Compute arithmetic series + diff

```py
class Solution:
    def minimalKSum(self, nums: List[int], k: int) -> int:
        def arith_sum(n, a):
            return n*(2*a+n-1)//2
        nums.append(0)
        nums.append(10**10)
        nums = sorted(list(set(nums)))
        res = 0
        for i in range(1,len(nums)):
            diff = nums[i]-nums[i-1]-1
            delta = min(k,diff)
            k-= delta
            res += arith_sum(delta, nums[i-1]+1)
            if k==0: break
        res += arith_sum(k, nums[-1]+1)
        return res
```

### Solution: binary search 

```py
class Solution:
    def minimalKSum(self, nums: List[int], k: int) -> int:
        nums = set(nums)
        lo, hi = 1, 10**10
        while lo<hi:
            mid = (lo+hi)>>1
            if mid - sum(i<=mid for i in nums) >= k:
                hi = mid
            else:
                lo = mid+1
        return lo*(lo+1)//2 - sum(i for i in nums if i<=lo)
```

## 2196. Create Binary Tree From Descriptions

### Solution: Hashmap + construct tree online

```py
class Solution:
    def createBinaryTree(self, descriptions: List[List[int]]) -> Optional[TreeNode]:
        nodes = {}
        children = set()
        for parent, child, is_left in descriptions:
            pnode = nodes.setdefault(parent, TreeNode(parent))
            cnode = nodes.setdefault(child, TreeNode(child))
            children.add(child)
            if is_left:
                pnode.left = cnode
            else:
                pnode.right = cnode
        for key, node in nodes.items():
            if key not in children:
                return node
        print("Error the description does not contain a valid tree with a root node")
        return -1
```

## 2197. Replace Non-Coprime Numbers in Array

### Solution: stack + gcd + lcm 

This must be a weak spot for me, cause I just thought use
doubly linked list when you need to delete elements, but I forget
you can use a stack. 

```py
from math import gcd, lcm
class Solution:
    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        stack = []
        for num in nums:
            while stack and gcd(stack[-1], num) > 1:
                num = lcm(stack[-1],num)
                stack.pop()
            stack.append(num)
        return stack
```

# Leetcode Weekly Contest 284

## 

### Solution: 

```py

```

## 

### Solution: 

```py

```

## 

### Solution: 

```py

```

## 2203. Minimum Weighted Subgraph With the Required Paths

### Solution: forward + reverse graph dijkstra algorithm

```py
from heapq import heappush, heappop

class Solution:
    def minimumWeight(self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int) -> int:
        Graphs = namedtuple('Graphs', ['forward', 'reverse'])
        graphs = Graphs([[] for _ in range(n)], [[] for _ in range(n)])
        for x, y, w in edges:
            graphs.forward[x].append((y,w))
            graphs.reverse[y].append((x,w))
            
        def dijkstra_forward(src):
            dist = defaultdict(lambda: math.inf)
            vis = set()
            heap = []
            heappush(heap, (0, src))
            dist[src] = 0
            while heap:
                cost, node = heappop(heap)
                if node in vis: continue
                for nei, nw in graphs.forward[node]:
                    vis.add(node)
                    ncost = cost + nw
                    if ncost < dist[nei]:
                        heappush(heap, (ncost, nei))
                        dist[nei] = ncost
            return dist
        
        def dijkstra_reverse(src):
            dist = defaultdict(lambda: math.inf)
            vis = set()
            heap = []
            heappush(heap, (0, src))
            dist[src] = 0
            while heap:
                cost, node = heappop(heap)
                if node in vis: continue
                for nei, nw in graphs.reverse[node]:
                    vis.add(node)                   
                    ncost = cost + nw
                    if ncost < dist[nei]:
                        heappush(heap, (ncost, nei))
                        dist[nei] = ncost
            return dist
        dest_dist = dijkstra_reverse(dest)
        if dest_dist[src1]==math.inf or dest_dist[src2]==math.inf: return -1
        src1_dist = dijkstra_forward(src1)
        src2_dist = dijkstra_forward(src2)
        best = math.inf
        for i in range(n):
            best = min(best, dest_dist[i]+src1_dist[i]+src2_dist[i])
        return best
```

Improved dijkstra implementation most likely

```py
from heapq import heappush, heappop

class Solution:
    def minimumWeight(self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int) -> int:
        Graphs = namedtuple('Graphs', ['forward', 'reverse'])
        graphs = Graphs(defaultdict(list), defaultdict(list))
        for x, y, w in edges:
            graphs.forward[x].append((y,w))
            graphs.reverse[y].append((x,w))
            
        def dijkstra_forward(src):
            dist = {}
            heap = []
            heappush(heap, (0, src))
            while heap:
                cost, node = heappop(heap)
                if node in dist: continue
                dist[node] = cost
                for nei, nw in graphs.forward[node]:
                    ncost = cost + nw
                    if ncost < dist.get(nei,math.inf):
                        heappush(heap, (ncost, nei))
            return dist
        
        def dijkstra_reverse(src):
            dist = {}
            heap = []
            heappush(heap, (0, src))
            while heap:
                cost, node = heappop(heap)
                if node in dist: continue
                dist[node] = cost
                for nei, nw in graphs.reverse[node]:                
                    ncost = cost + nw
                    if ncost < dist.get(nei, math.inf):
                        heappush(heap, (ncost, nei))
            return dist
        dest_dist = dijkstra_reverse(dest)
        if src1 not in dest_dist or src2 not in dest_dist: return -1
        src1_dist = dijkstra_forward(src1)
        src2_dist = dijkstra_forward(src2)
        best = math.inf
        for i in range(n):
            best = min(best, dest_dist.get(i, math.inf)+src1_dist.get(i, math.inf)+src2_dist.get(i, math.inf))
        return best
```

For Numba experiment try this testcase 

5

# Leetcode Weekly Contest 286

## Summary

I struggled a little too much on Q3,  I just did not know the easy way to 
generate a palindrome for integers in order,  I know you can consider just the
first half of the palindrome, since it is reflected.  But I didn't know that if you
look at 100, 101, 102, 103, 104, 105, 106, ... 190, these are 90 palindromes and when you reflect
you get either two depending if odd or even length. 

My struggle on Q4 was to optimize the dynamic programming solution.  I think my solution is about
O(k^3), but not sure on taht. 

## 2215. Find the Difference of Two Arrays

### Solution: set theory, difference between two sets

```py
class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        return [set(nums1)-set(nums2), set(nums2)-set(nums1)]
```


## 2216. Minimum Deletions to Make Array Beautiful

```py
class Solution:
    def minDeletion(self, nums: List[int]) -> int:
        n = len(nums)
        i = cnt = 0
        while i < len(nums)-1:
            if nums[i]==nums[i+1]:
                i += 1
                cnt += 1
            else:
                i += 2
        return cnt + ((n-cnt)%2==1)
```

```py
class Solution:
    def minDeletion(self, nums: List[int]) -> int:
        answer = []
        for num in nums:
            if len(answer)%2==0 or answer[-1]!=num:
                answer.append(num)
        return len(nums)-(len(answer)-len(answer)%2)
```

## 2217. Find Palindrome With Fixed Length

```py
class Solution:
    def kthPalindrome(self, queries: List[int], intLength: int) -> List[int]:
        base = 10**((intLength-1)//2)
        result = [base + q - 1 for q in queries]
        answer = [-1]*len(queries)
        for i, pal in enumerate(result):
            if intLength%2==0:
                spal = str(pal) + str(pal)[::-1]
            else:
                spal = str(pal) + str(pal)[:-1][::-1]
            if len(spal)==intLength:
                answer[i] = int(spal)
        return answer
```


## 2218. Maximum Value of K Coins From Piles

### Solution:  recursive dp that TLE

```py
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        num_piles = len(piles)
        # too slow it is O(n*k*k)
        def dfs(current_pile, i, num_coins):
            if num_coins == k or current_pile==num_piles: return 0
            if i==len(piles[current_pile]):
                return dfs(current_pile+1,0,num_coins)
            return max(dfs(current_pile+1,0,num_coins), dfs(current_pile, i+1,num_coins+1) + piles[current_pile][i])
        return dfs(0,0,0)
```

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
        wins, loss = set(), Counter()
        for u, v in matches:
            wins.add(u)
            loss[v] += 1
        return [sorted(wins - set(loss)), sorted([x for x, v in loss.items() if v == 1])]
```

## 2226. Maximum Candies Allocated to K Children
 
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

# Leetcode Weekly Contest 289

## Summary

## 2243. Calculate Digit Sum of a String

### Solution 1: generator to yield the sum of every k digits 

```py
class Solution:
    def digitSum(self, s: str, k: int) -> str:
        def get_digit_sum(digits):
            for i in range(0,len(digits),k):
                yield sum(map(int, digits[i:i+k]))
        while len(s) > k:
            s = "".join(map(str, get_digit_sum(s)))
        return s
```

## 2244. Minimum Rounds to Complete All Tasks

### Solution 1: Counter + hash table

```py
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        counter = Counter(tasks)
        if any(cnt==1 for cnt in counter.values()):
            return -1
        return sum(cnt//3 if cnt%3==0 else cnt//3+1 for cnt in counter.values())
```

## 2245. Maximum Trailing Zeros in a Cornered Path

### Solution 1: prefix sum with every pair 2 and 5 contributes to a trailing zero

```py
class Solution:
    def maxTrailingZeros(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        hor_prefix = [[[0,0] for _ in range(C+1)] for _ in range(R)]
        vert_prefix = [[[0,0] for _ in range(C)] for _ in range(R+1)]
        for r, c in product(range(R), range(C)):
            elem = grid[r][c]
            cnt2 = cnt5 = 0
            while elem > 0 and elem%2==0:
                cnt2 += 1
                elem//=2
            while elem > 0 and elem%5==0:
                cnt5 += 1
                elem//=5
            hor_prefix[r][c+1][0] = hor_prefix[r][c][0] + cnt2
            hor_prefix[r][c+1][1] = hor_prefix[r][c][1] + cnt5
            vert_prefix[r+1][c][0] = vert_prefix[r][c][0] + cnt2
            vert_prefix[r+1][c][1] = vert_prefix[r][c][1] + cnt5
        max_zeros = 0
        def pair(A, B):
            return min(A[0]+B[0], A[1]+B[1])
        for r, c in product(range(R), range(C)):
            right, left= list(map(lambda x: x[0]-x[1], zip(hor_prefix[r][C],hor_prefix[r][c]))), hor_prefix[r][c]
            bottom, top = list(map(lambda x: x[0]-x[1], zip(vert_prefix[R][c],vert_prefix[r+1][c]))), vert_prefix[r][c]
            # print(left, right, bottom, top)
            max_zeros = max(max_zeros, pair(left, top), pair(left, bottom), pair(right, top), pair(right, bottom))
        return max_zeros
```

## Solution 2: Numpy + np.cumsum + np.minimum + np.rot90

```py
import numpy as np
class Solution:
    def maxTrailingZeros(self, grid: List[List[int]]) -> int:
        A = np.array(grid)
        def prefix_sums(digit):
            sa = sum(A%digit**i==0 for i in range(1,10))
            return np.cumsum(sa, axis=0) + np.cumsum(sa, axis=1) - sa
        return max(np.minimum(prefix_sums(2), prefix_sums(5)).max() 
                  for _ in range(4) if bool([A := np.rot90(A)]))
```

## 2246. Longest Path With Different Adjacent Characters

### Solution 1:

```py

```

# Leetcode Weekly Contest 290

## Summary

## 2248. Intersection of Multiple Arrays

### Solution 1:  set intersection

```py
class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        intersect_set = set(nums[0])
        for i in range(1,len(nums)):
            intersect_set &= set(nums[i])
        return sorted(intersect_set)
```

```py
class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        return sorted(set.intersection(*[set(nums[i]) for i in range(len(nums))]))
```

## 2249. Count Lattice Points Inside a Circle

### Solution 1: brute force with reduced search space

```py

class Solution:
    def countLatticePoints(self, circles: List[List[int]]) -> int:
        cnt = 0
        maxx = max(x+r for x, _, r in circles)
        maxy = max(y+r for _, y, r in circles)
        minx = min(x-r for x, _, r in circles)
        miny = min(y-r for _,y,r in circles)
        for x, y in product(range(minx, maxx+1), range(miny,maxy+1)):
            for xc, yc, r in circles:
                if math.hypot(abs(x-xc),abs(y-yc)) <= r:
                    cnt+=1
                    break
        return cnt
```

## 2250. Count Number of Rectangles Containing Each Point

### Solution 1: sort + binary search on the large width, and iterate through the small heights, height << width

```py

class Solution:
    def countRectangles(self, rectangles: List[List[int]], points: List[List[int]]) -> List[int]:
        n=len(points)
        counts = [0]*n
        rects = defaultdict(list)
        for l, h in rectangles:
            rects[h].append(l)
        heights = sorted([h for h in rects.keys()])
        for h in heights:
            rects[h].sort()
        for i, (x, y) in enumerate(points):
            hstart = bisect_left(heights, y)
            for j in range(hstart, len(heights)):
                cur_h = heights[j]
                k = len(rects[cur_h]) - bisect_left(rects[cur_h], x)
                counts[i] += k
        return counts
```

## 2251. Number of Flowers in Full Bloom

### Solution 1: line sweep with pointer for persons + sort

```py
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], persons: List[int]) -> List[int]:
        n = len(persons)
        persons = sorted([(p, i) for i, p in enumerate(persons)])
        answer = [0]*n
        events = []
        for s, e in flowers:
            events.append((s, 1))
            events.append((e+1,-1))
        events.sort()
        i = 0
        bloomed = 0 
        for ev, delta in events:
            while i < n and persons[i][0] < ev:
                p, index  = persons[i]
                answer[index] = bloomed
                i+=1
            bloomed += delta
            if i == n: break
        return answer
        
```

### Solution 2: two binary search for start and end

```py
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], persons: List[int]) -> List[int]:
        start, end = sorted(s for s, e in flowers), sorted(e for s,e in flowers)
        return [bisect_right(start,time) - bisect_left(end, time) for time in persons]
```

# Leetcode Weekly Contest 291

## Summary

## 2259. Remove Digit From Number to Maximize Result

### Solution 1: math + greedy, choose the first time you see digit when if you remove it, it will be replaced by larger digit to immediate left of it. 

```py
class Solution:
    def removeDigit(self, number: str, digit: str) -> str:
        idx = 0
        for i, s in enumerate(number):
            if s==digit:
                idx = i
                if i+1 < len(number) and number[i+1] > digit:
                    break
        return number[:idx] + number[idx+1:]
```

## 2260. Minimum Consecutive Cards to Pick Up

### Solution 1: hash table to store last index of card

```py
class Solution:
    def minimumCardPickup(self, cards: List[int]) -> int:
        last_idx = {}
        best = inf
        for i, c in enumerate(cards):
            if c in last_idx:
                best = min(best, i - last_idx[c] + 1)
            last_idx[c] = i
        return best if best < inf else -1
```

## 2261. K Divisible Elements Subarrays

### Solution 1: hash table to store the subarrays

```py
class Solution:
    def countDistinct(self, nums: List[int], k: int, p: int) -> int:
        cntArrays = 0
        n = len(nums)
        seen = set()
        for i in range(n):
            cntDiv = 0
            subarray = []
            for j in range(i,n):
                cntDiv += (nums[j]%p==0)
                if cntDiv > k: break
                subarray.append(nums[j])
                hash_ = tuple(subarray)
                if hash_ in seen: continue
                cntArrays += 1
                seen.add(hash_)
        return cntArrays
```

## 2262. Total Appeal of A String

### Solution 1: Store current delta, and hash table for last index

For each character, you add it to all the previous substrings consider, and so you will add that appeal delta.  but also you want to increase the appeal for all the substrings that will gain an increase of one appeal.  Which will be basically the number of substrings found based on the last index location of this current character.  So you add that to the delta, because it is the new delta for the appeal that we add at each additional character. 

```py
class Solution:
    def appealSum(self, s: str) -> int:
        last_idx = defaultdict(lambda: -1)
        delta = sum_ = 0
        for i, ch in enumerate(s):
            delta += (i-last_idx[ch])
            sum_ += delta
            last_idx[ch] = i
        return sum_
```

# Leetcode Weekly Contest 291

## Summary

##

###

```py

```

##

###

```py

```

##

###

```py

```

##

###

```py

```

# Leetcode Weekly Contest 337


## 2595. Number of Even and Odd Bits

### Solution 1:  enumerate through binary representation + track index

```py
class Solution:
    def evenOddBit(self, n: int) -> List[int]:
        res = [0]*2
        for i, dig in enumerate(reversed(bin(n)[2:])):
            if dig == '1':
                res[i&1] += 1
        return res
```

## 2596. Check Knight Tour Configuration

### Solution 1:  hash table + check valid move by min needs be 1 and max needs be 2 amongst two values

```py
class Solution:
    def checkValidGrid(self, grid: List[List[int]]) -> bool:
        n = len(grid)
        if grid[0][0] != 0: return False
        cell_loc = {}
        for r, c in product(range(n), repeat = 2):
            cell_loc[grid[r][c]] = (r, c)
        for i in range(1, n*n):
            pr, pc = cell_loc[i - 1]
            r, c = cell_loc[i]
            dr, dc = abs(r - pr), abs(c - pc)
            if min(dr, dc) != 1 or max(dr, dc) != 2: return False
        return True
```

## 2597. The Number of Beautiful Subsets

### Solution 1:  sort + counter + backtrack + recursion

```py
class Solution:
    def beautifulSubsets(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        counts = [0]*1001
        cnt = 0
        def backtrack(i):
            nonlocal cnt
            if i == n: return int(cnt > 0)
            res = 0
            res += backtrack(i + 1) # skip
            prv = nums[i] - k
            if counts[prv] == 0:
                counts[nums[i]] += 1
                cnt += 1
                res += backtrack(i + 1)
                counts[nums[i]] -= 1
                cnt -= 1
            return res
        return backtrack(0)
```

## 2598. Smallest Missing Non-negative Integer After Operations

### Solution 1:  modular arithmetic + min with custom key

Find the minimum cnt that is the number of times you can wrap around the values. And then the min_idx is how far get on last wrap around so if wrap around once min_cnt is 1, and there might be min_idx of 2 or something so answer is 1*value + 2

```py
class Solution:
    def findSmallestInteger(self, nums: List[int], value: int) -> int:
        counts = [0]*value
        for v in map(lambda num: num%value, nums):
            counts[v] += 1
        min_idx, min_cnt = min(enumerate(counts), key = lambda pair: (pair[1], pair[0]))
        return value*min_cnt + min_idx
```

# Leetcode Weekly Contest 339

## 2609. Find the Longest Balanced Substring of a Binary String

### Solution 1:  groupby to get the 0s and 1s together + use the min of the groups

```py
class Solution:
    def findTheLongestBalancedSubstring(self, s: str) -> int:
        res = cur = 0
        for key, grp in groupby(s):
            cnt = len(list(grp))
            if key == '0':
                cur = cnt
            else:
                cur = min(cur, cnt)
                res = max(res, 2*cur)
        return res
```

## 2610. Convert an Array Into a 2D Array With Conditions

### Solution 1:  implementation

```py
class Solution:
    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        last = [0] * (n + 1)
        ans = []
        for num in nums:
            if len(ans) == last[num]:
                ans.append([])
            ans[last[num]].append(num)
            last[num] += 1
        return ans              
```

## 2611. Mice and Cheese

### Solution 1:  sorted + difference array

The idea to give the ones with largest difference is because the difference is calculating when you get the most value from having mouse 1 eat the cheese at index i.  So it makes sense to give it to mouse 1, since you only going to give k, then give the rest to mouse 2. 

```py
class Solution:
    def miceAndCheese(self, reward1: List[int], reward2: List[int], k: int) -> int:
        n = len(reward1)
        diff = sorted([(r1 - r2, i) for i, (r1, r2) in enumerate(zip(reward1, reward2))], reverse = True)
        res = 0
        vis = [0]*n
        for i in range(k):
            index = diff[i][1]
            vis[index] = 1
            res += reward1[index]
        return res + sum([reward2[i] for i in range(n) if not vis[i]])
```

### Solution 2:  nlargest + zip + sum

Observe that you only need the k largest elements from the difference array and add them to the sum of giving all of it to mouse 2.

```py
class Solution:
    def miceAndCheese(self, reward1: List[int], reward2: List[int], k: int) -> int:
        return sum(reward2) + sum(nlargest(k, (r1 - r2 for r1, r2 in zip(reward1, reward2))))
        
```

## 2612. Minimum Reverse Operations

### Solution 1:  bfs + sortedlist + parity + O(nlogn) 

Basic idea is that whenever you are given a range of index, you will take increments of two

This is the pattern for the right side, but note it is symmetric so it does the same moving to the left of the current index as well. 

But basically the observation is if you are at i, you increment by 2, so therefore just need to figure out parity and then go through available nodes

k
1  i
2  i+1
3  i, i+2
4  i+1, i+3
5  i, i+2, i+4
6  i+1, i+3, i+5
7  i, i+2, i+4, i+6

To store available nodes put them in two sortedlist so that it follows the parity, so all index even in 0th element and all index odd in 1st element, which is just adding them to a sortedlist.  Reason for sortedlist is so that can remove elements from it in O(logn) time.  This is way to prevent rechecking already visited nodes. No reason to revisit, already found minimun reverse operations.

You begin a bfs from the current p.

Then for each pos you find the left and right bounds, which is tricky to derive, but write it down on a piece of paper and you can derive it.  

for instance k = 4 you have

so looking at the 1,2,3,4 step in here, you have these bounds that are need to move 1 to p-3 position or to p + 3 position and so on. 

4 [p, p+3] p+3
3 [p-1, p+2] p+1 
2 [p-2, p+1] p-1
1 [p-3, p] p-3

Now consider a formula to derive the left bounds for the index, is it p-3, or p-1, or p+1

Well if you push it back as far as you can so pos - k + 1, this will get it to as far back it can be, 
Then there is a pattern of increment based on the left point being all the way over there

p    p+3  +3
p-1  p+1  +2
p-2  p-1  +1
p-3  p-3  +0

see there is a  pattern if you push it all the way back, and it is basically that it the different increments by 1 for each iteration farther away

so for example if p+1 should be left bound, and I've pushed left to p-1 then I need to do p-1 + 2, but how can I find 2

it can be found by taking k - 1 + left - pos, cause basically k - 1 + p - 1 - p = k - 1 - 1 = 4 - 2 = 2, which is correct, and think about it, basically as left decreases, the delta becomes

so increment is k - 1 + left - pos so you can find current left position for the neighbors by taking left + increment

In addition you can find right pointer in similar manner, just take 
right = min(n - 1, pos + k - 1) - k + 1, so basically when you add the k it will be the right point above, but so you can use the same formula as above, just move it back k - 1. 

This way it follows the same pattern above, and can be calcualted in same way, and you can find the appropriate max. 

but basically you find the leftmost you can go, and the rightmost, cause that is all it can visit.

```py
from sortedcontainers import SortedList

class Solution:
    def minReverseOperations(self, n: int, p: int, banned: List[int], k: int) -> List[int]:
        nodes = [SortedList(), SortedList()]
        banned = set(banned)
        for i in range(n):
            if i == p or i in banned: continue
            nodes[i%2].add(i)
        queue = deque([p])
        dist = [-1]*n
        dist[p] = 0
        while queue:
            pos = queue.popleft()
            left = max(0, pos - k + 1)
            left = 2*left + k - 1 - pos
            right = min(n - 1, pos + k - 1) - k + 1
            right = 2*right + k - 1 - pos
            used = []
            for nei_pos in nodes[left%2].irange(left, right):
                dist[nei_pos] = dist[pos] + 1
                queue.append(nei_pos)
                used.append(nei_pos)
            for i in used:
                nodes[left%2].remove(i)
        return dist
```

# Leetcode Weekly Contest 340

## 2614. Prime In Diagonal

### Solution 1:  matrix + prime + math + O(sqrt(n)) primality check

```py
class Solution:
    def diagonalPrime(self, nums: List[List[int]]) -> int:
        n = len(nums)
        memo = {}
        def is_prime(x: int) -> bool:
            if x in memo: return memo[x]
            if x < 2: return False
            for i in range(2, int(math.sqrt(x)) + 1):
                if x % i == 0: return False
            return True
        res = 0
        for i in range(n):
            if is_prime(nums[i][i]):
                res = max(res, nums[i][i])
            if is_prime(nums[i][~i]):
                res = max(res, nums[i][~i])
        return res
```

### Solution 2:  Sieve of Eratosthenes + precompute primality

```py

```

## 2615. Sum of Distances

### Solution 1:  prefix + suffix sums and counts + line sweep + hash table

```py
class Solution:
    def distance(self, nums: List[int]) -> List[int]:
        n = len(nums)
        last_index = Counter()
        suffix = Counter()
        suffix_cnt = Counter()
        prefix, pcnter = Counter(), Counter()
        for i, num in enumerate(nums):
            suffix[num] += i
            suffix_cnt[num] += 1
        ans = [0]*n
        for i, num in enumerate(nums):
            delta = i - last_index[num]
            suffix[num] -= delta*suffix_cnt[num]
            prefix[num] += delta*pcnter[num]
            ans[i] = prefix[num] + suffix[num]
            suffix_cnt[num] -= 1
            pcnter[num] += 1
            last_index[num] = i
        return ans
```

## 2616. Minimize the Maximum Difference of Pairs

### Solution 1:  greedy binary search

count every other greedily to check if it has enough pais where it is less than or equal to target.  But just know you have to move iterator two forward, so there is no overlap

```py
class Solution:
    def minimizeMax(self, nums: List[int], p: int) -> int:
        if p == 0: return 0
        n = len(nums)
        nums.sort()
        def possible(target):
            cnt = 0
            i = 1
            while i < n:
                if nums[i] - nums[i - 1] <= target:
                    cnt += 1
                    i += 1
                i += 1
            return cnt >= p
        left, right = 0, nums[-1] - nums[0]
        while left < right:
            mid = (left + right) >> 1
            if not possible(mid):
                left = mid + 1
            else:
                right = mid
        return left
```

## 2617. Minimum Number of Visited Cells in a Grid

### Solution 1:  bfs with boundary or frontier optimization

```py
class Solution:
    def minimumVisitedCells(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        # FRONTIERS
        max_row, max_col = [0]*C, [0]*R
        queue = deque([(0, 0)])
        dist = 1
        while queue:
            for _ in range(len(queue)):
                r, c = queue.popleft()
                if (r, c) == (R - 1, C - 1): return dist
                # RIGHTWARD MOVEMENT    
                for k in range(max(c, max_col[r]) + 1, min(grid[r][c] + c, C - 1) + 1):
                    queue.append((r, k))
                # DOWNWARD MOVEMENT
                for k in range(max(r, max_row[c]) + 1, min(grid[r][c] + r, R - 1) + 1):
                    queue.append((k, c))
                # UPDATE FRONTIERS
                max_col[r] = max(max_col[r], grid[r][c] + c)
                max_row[c] = max(max_row[c], grid[r][c] + r)
            dist += 1
        return -1
```

### Solution 2:  sortedlist to track non visited nodes + irange to quickly find next nodes

```py
from sortedcontainers import SortedList
class Solution:
    def minimumVisitedCells(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        rows, cols = [SortedList(range(R)) for _ in range(C)], [SortedList(range(C)) for _ in range(R)]
        queue = deque([(0, 0)])
        dist = 1
        while queue:
            for _ in range(len(queue)):
                r, c = queue.popleft()
                if (r, c) == (R - 1, C - 1): return dist
                # RIGHTWARD MOVEMENT
                for k in list(cols[r].irange(c + 1, grid[r][c] + c)):
                    queue.append((r, k))
                    cols[r].remove(k)
                    rows[k].remove(r)
                # DOWNWARD MOVEMENT
                for k in list(rows[c].irange(r + 1, grid[r][c] + r)):
                    queue.append((k, c))
                    rows[c].remove(k)
                    cols[k].remove(c)
            dist += 1
        return -1
```

# Leetcode Weekly Contest 341

## 2643. Row With Maximum Ones

### Solution 1:  one liner with max and custom comparator

```py
class Solution:
    def rowAndMaximumOnes(self, mat: List[List[int]]) -> List[int]:
        return max([[sum(row), r] for r, row in enumerate(mat)], key = lambda elem: (elem[0], -elem[1]))[::-1]
```

## 2644. Find the Maximum Divisibility Score

### Solution 1:  max + tiebreaker minimize on second element + maximum applied to tuples

```py
class Solution:
    def maxDivScore(self, nums: List[int], divisors: List[int]) -> int:
        result = (-math.inf, -math.inf)
        for div in divisors:
            div_score = sum([1 for num in nums if num%div == 0])
            result = max(result, (div_score, div), key = lambda pair: (pair[0], -pair[1]))
        return result[1]
```

## 2645. Minimum Additions to Make Valid String

### Solution 1:  cycle matching

```py
class Solution:
    def addMinimum(self, word: str) -> int:
        target = "abc"
        res, n, j= 0, len(word), 0
        for i in range(n):
            while word[i] != target[j]:
                res += 1
                j = (j + 1)%len(target)
            j = (j + 1)%len(target)
        return res + (3 - j if j > 0 else 0)
```

### Solution 2:  counting number of "abc" strings

```py
class Solution:
    def addMinimum(self, word: str) -> int:
        cycles = 0
        n = len(word)
        prev = 'z'
        for i in range(n):
            cycles += word[i] <= prev
            prev = word[i]
        return 3*cycles - n
```

## 2646. Minimize the Total Price of the Trips

### Solution 1:  dfs + dynammic programming on tree + path reconstruction

1. build adjacency list
1. construct a frequency array for each shortest path between the node pairs in trips.  can use a iterative dfs algorithm and store the parent nodes along the way to be able to pass back through from end to start node along the shortest path in a tree and compute the frequency of each node.
1. dynammic programming with the states being the current node and if the previous node price was halved or not.  get the minimum of these two options.

```py
class Solution:
    def minimumTotalPrice(self, n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:
        # CONSTRUCT ADJACENCY LIST
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        # BUILD FREQUENCY ARRAY WITH DFS
        freq = [0]*n
        for start, end in trips:
            stack = [(start, -1)]
            parent_arr = [-1]*n
            while stack:
                node, parent = stack.pop()
                parent_arr[node] = parent
                if node == end: break
                for nei in adj_list[node]:
                    if nei == parent: continue
                    stack.append((nei, node))
            # GO THROUGH PATH
            while node != -1:
                freq[node] += 1
                node = parent_arr[node]
        # DYNAMMIC PROGRAMMING ON ARBITRARY ROOT OF TREE
        @cache
        def dp(node, parent, prev_halved):
            halved_sum = math.inf if prev_halved else 0
            full_sum = 0
            for nei in adj_list[node]:
                if nei == parent: continue
                full_sum += dp(nei, node, 0)
                if not prev_halved:
                    halved_sum += dp(nei, node, 1)
            return min(halved_sum + price[node]*freq[node]//2, full_sum + price[node]*freq[node])
        return dp(0, -1, 0)
```

# Leetcode Weekly Contest 342

## 2651. Calculate Delayed Arrival Time

### Solution 1:  modulus

```py
class Solution:
    def findDelayedArrivalTime(self, arrivalTime: int, delayedTime: int) -> int:
        return (arrivalTime + delayedTime)%24
```

## 2652. Sum Multiples

### Solution 1:  sum + loop + modulus

```py
class Solution:
    def sumOfMultiples(self, n: int) -> int:
        return sum(i for i in range(1, n + 1) if i%3 == 0 or i%5 == 0 or i%7 == 0)
```

## 2653. Sliding Subarray Beauty

### Solution 1:  constant size sliding window + sortedlist

```py
from sortedcontainers import SortedList
class Solution:
    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        window = SortedList()
        res = []
        for i in range(n):
            window.add(nums[i])
            if len(window) == k:
                res.append(min(0, window[x - 1]))
                window.remove(nums[i - k + 1])
        return res
```

### Solution 2: constant size sliding window + frequency of negative integers + O(50n)

```py
class Solution:
    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        freq, res = [0]*50, []
        for i in range(n):
            if nums[i] < 0: freq[nums[i] + 50] += 1
            if i >= k and nums[i - k] < 0: freq[nums[i - k] + 50] -= 1
            if i >= k - 1:
                cnt = 0
                for j in range(50):
                    cnt += freq[j]
                    if cnt >= x:
                        res.append(j - 50)
                        break
                if cnt < x: res.append(0)
        return res
```

## 2654. Minimum Number of Operations to Make All Array Elements Equal to 1

### Solution 1:  smallest gcd subarray equal to 1 + O(n^2)

```py
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ones = nums.count(1)
        if ones: return n - ones
        res = math.inf
        for i in range(n):
            gcd_prefix = nums[i]
            for j in range(i + 1, n):
                gcd_prefix = math.gcd(gcd_prefix, nums[j])
                if gcd_prefix == 1:
                    res = min(res, j - i)
                    break
        return res + n - 1 if res != math.inf else -1
```

# Leetcode Weekly Contest 343

## 2660. Determine the Winner of a Bowling Game

### Solution 1: loop

```py
class Solution:
    def isWinner(self, player1: List[int], player2: List[int]) -> int:
        def score(player):
            prev = sum_ = 0
            for x in player:
                sum_ += x
                if prev > 0:
                    sum_ += x
                    prev -= 1
                if x == 10:
                    prev = 2
            return sum_
        sum1, sum2 = map(score, (player1, player2))
        if sum1 == sum2: return 0
        return 1 if sum1 > sum2 else 2
```

## 2661. First Completely Painted Row or Column

### Solution 1:  hash table + horizontal and vertical sum

```py
class Solution:
    def firstCompleteIndex(self, arr: List[int], mat: List[List[int]]) -> int:
        R, C = len(mat), len(mat[0])
        n = len(arr)
        horz_sum, vert_sum = [0]*R, [0]*C
        pos = {mat[r][c]: (r, c) for r, c in product(range(R), range(C))}
        for i, val in enumerate(arr):
            r, c = pos[val]
            horz_sum[r] += 1
            vert_sum[c] += 1
            if horz_sum[r] == C or vert_sum[c] == R: return i
        return n
```

## 2662. Minimum Cost of a Path With Special Roads

### Solution 1:  bfs + memoization + shortest path in directed graph

Find the minimum cost to travel to each end node in the specialRoads, so sometimes can use multiple specialRoads to get there.  So using bfs to build up this path.  And relax the cost of each node in the path when you find a better route.

Then treat each end location in the spcialRoads as a start point for getting to the target. 

```py
class Solution:
    def minimumCost(self, start: List[int], target: List[int], specialRoads: List[List[int]]) -> int:
        E = len(specialRoads)
        min_cost = defaultdict(lambda: math.inf)
        queue = deque([tuple(start)])
        min_cost[tuple(start)] = 0
        manhattan_distance = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        while queue:
            x, y = queue.popleft()
            for x1, y1, x2, y2, cost in specialRoads:
                ncost = min_cost[(x, y)] + manhattan_distance(x, y, x1, y1) + cost
                if ncost < min_cost[(x2, y2)]:
                    min_cost[(x2, y2)] = ncost
                    queue.append((x2, y2))
        res = math.inf
        for (x, y), cost in min_cost.items():
            cur_cost = cost + manhattan_distance(*target, x, y) 
            res = min(res, cur_cost)
        return res
```

##

### Solution 1:

```py

```



# Leetcode Weekly Contest 344

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



# Leetcode Weekly Contest 345

## 2682. Find the Losers of the Circular Game

### Solution 1:  simulate

```py
class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        winners = [0]*n
        friend = 0
        turn = 1
        while True:
            if winners[friend]: break
            winners[friend] = 1
            friend = (friend + turn*k)%n
            turn += 1
        return [i + 1 for i in range(n) if not winners[i]]
```

## 2683. Neighboring Bitwise XOR

### Solution 1:  simulate 

two options either start with 0 or 1 in the array, that will determine the rest of the values for the original array. if first and last element are equal then it was perfectly circular and works. 

```py
from typing import *
class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        derived.append(derived[0])
        for start in range(2):
            arr = [0]*(n + 1)
            arr[0] = start
            for i in range(n):
                if derived[i] == 0:
                    arr[i + 1] = arr[i]
                else:
                    arr[i + 1] = arr[i] ^ 1
            if arr[0] == arr[-1]: return True
        return False
```

### Solution 2:  xor sum

Observe that the xor of the derived array is equal to 0 indicates the original array can be formed. 

![xor_derivation](images/neighboring_bitwise_xor.png)

```py
class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        return reduce(operator.xor, derived) == 0
```

## 2684. Maximum Number of Moves in a Grid

### Solution 1:  iterative dynamic programming

```py
class Solution:
    def maxMoves(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        dp = [[-math.inf]*C for _ in range(R)]
        # base case initialize to 0 for all 0 column elements
        for r in range(R):
            dp[r][0] = 0
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        res = 0
        for c in range(1, C):
            for r in range(R):
                for nr, nc in [(r + 1, c - 1), (r, c - 1), (r - 1, c - 1)]:
                    if not in_bounds(nr, nc) or grid[r][c] <= grid[nr][nc]: continue
                    dp[r][c] = max(dp[r][c], dp[nr][nc] + 1)
                res = max(res, dp[r][c])
        return res
```

## 2685. Count the Number of Complete Components

### Solution 1:  union find + graph theory + math

mathematical relationship between vertex and edge count in a completed graph

A complete graph has v(v-1)/2 edges

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
        self.edge_count = [0]*n
    
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
        self.edge_count[i] += 1
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            self.edge_count[i] += self.edge_count[j]
            return True
        return False
    
class Solution:
    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        f = lambda x: (x - 1)*x//2
        dsu = UnionFind(n)
        for u, v in edges:
            dsu.union(u, v)
        roots = [0]*n
        res = 0
        for i in range(n):
            root = dsu.find(i)
            if roots[root]: continue
            roots[root] = 1
            vertex_count = dsu.size[root]
            edge_count = dsu.edge_count[root]
            res += f(vertex_count) == edge_count
        return res
```

### Solution 2:  dfs with stack + connected components

Increment count of complete connected components for when all nodes in the component have count of neighbors equals number of nodes minus 1.

```py
class Solution:
    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        vis = [0]*n
        res = 0
        for i in range(n):
            if vis[i]: continue
            nodes = [i]
            stack = [i]
            vis[i] = 1
            while stack:
                node = stack.pop()
                for nei in adj_list[node]:
                    if vis[nei]: continue
                    vis[nei] = 1
                    stack.append(nei)
                    nodes.append(nei)
            res += all(len(adj_list[node]) == len(nodes) - 1 for node in nodes)
        return res
```



# Leetcode Weekly Contest 346

## 2696. Minimum String Length After Removing Substrings

### Solution 1:  string slicing

```py
class Solution:
    def minLength(self, s: str) -> int:
        a, b = "AB", "CD"
        while a in s or b in s:
            if a in s:
                i = s.index(a)
                s = s[:i] + s[i + 2:]
            elif b in s:
                i = s.index(b)
                s = s[:i] + s[i + 2:]
        return len(s)
```

## 2697. Lexicographically Smallest Palindrome

### Solution 1:  two pointers + reverse second half string

```py
class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        left_s = s[:n//2]
        right_s = s[n//2 + (1 if n&1 else 0):][::-1]
        s_pal = []
        for i in range(n//2):
            if left_s[i] <= right_s[i]:
                s_pal.append(left_s[i])
            elif left_s[i] > right_s[i]:
                s_pal.append(right_s[i])
        if n & 1:
            s_pal.append(s[n//2])
        for i in range(n//2-1, -1, -1):
            s_pal.append(s_pal[i])
        return ''.join(s_pal)
```

## 2698. Find the Punishment Number of an Integer

### Solution 1:  bit mask

Try all partitions of the integer and check if it works or not.

```py
class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_valid(i):
            s = str(i * i)
            m = len(s)
            for mask in range(0, 1 << m):
                cur = total = 0
                for j in range(m):
                    if (mask >> j) & 1:
                        total += cur
                        cur = int(s[j])
                    else:
                        cur = cur * 10 + int(s[j])
                total += cur
                if total == i: 
                    return True
            return False
        return sum(i * i for i in range(1, n + 1) if is_valid(i))
```

## 2699. Modify Graph Edge Weights

### Solution 1:  dijkstra + reconstruct path

You have to be careful here because there is a reason this line is needed `edge_weights[edge] = max(target - rev_dists[v] - walked, 1)`.  It guarantees that the path picked as that with minimum distance from source to destination.  If you don't do this you could allow another path to become the path with minimum distance, and could lead to a minimum distance less than target.

This image shows the specific edge case where you need to give specific value at the -1 edge.  If you give edge weight of 4 it guarantees current path is still shortest path, if you were to give it something less than 4, there would be a different path that is shortest. if you give it greater than 4, just need to balance that fact later. 

![image](images/modified_edge_weights.PNG)

```py
class Solution:
    def modifiedGraphEdges(self, n: int, edges: List[List[int]], source: int, destination: int, target: int) -> List[List[int]]:
        adj_list = [[] for _ in range(n)]
        lower_bound, upper_bound = 1, 2*10**9
        for u, v, w in edges:
            adj_list[u].append((v, w))
            adj_list[v].append((u, w))
        def dijkstra(src, skip_mod):
            minheap = [(0, src)]
            dist = [math.inf] * n
            dist[src] = 0
            parent = {src: None}
            while minheap:
                cost, node = heapq.heappop(minheap)
                if cost > dist[node]: continue
                for nei, wei in adj_list[node]:
                    if wei == -1:
                        if skip_mod: continue
                        wei = lower_bound
                    ncost = cost + wei
                    if ncost < dist[nei]:
                        dist[nei] = ncost
                        heapq.heappush(minheap, (ncost, nei))
                        parent[nei] = node
            return dist, parent
        rev_dists, _ = dijkstra(destination, True)
        if rev_dists[source] < target: return []
        dists, parent = dijkstra(source, False)
        if dists[destination] > target: return []
        edge_weights = {(min(u, v), max(u, v)): w for u, v, w in edges}
        path = [destination]
        while path[-1] != source:
            path.append(parent[path[-1]])
        path = path[::-1]
        walked = 0
        for i in range(1, len(path)):
            u, v = path[i-1], path[i]
            edge = (min(u, v), max(u, v))
            if edge_weights[edge] == -1:
                edge_weights[edge] = max(target - rev_dists[v] - walked, 1)
            walked += edge_weights[edge]
        for u, v, w in edges:
            edge = (min(u, v), max(u, v))
            if edge_weights[edge] == -1:
                edge_weights[edge] = upper_bound
        return [[u, v, w] for (u, v), w in edge_weights.items()]
```




# Leetcode Weekly Contest 347

## 2710. Remove Trailing Zeros From a String

### Solution 1:  string

```py
class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        x = int(num[::-1])
        return str(x)[::-1]
```

## 2711. Difference of Number of Distinct Values on Diagonals

### Solution 1:  matrix hash with r - c key + counter for top and bottom diagonals for each diagonal

```py
class Solution:
    def differenceOfDistinctValues(self, grid: List[List[int]]) -> List[List[int]]:
        R, C = len(grid), len(grid[0])
        top_diags, bot_diags = defaultdict(Counter), defaultdict(Counter)
        for r, c in product(range(R), range(C)):
            bot_diags[r - c][grid[r][c]] += 1
        ans = [[0]*C for _ in range(R)]
        for r, c in product(range(R), range(C)):
            v = grid[r][c]
            bot_diags[r - c][v] -= 1
            if bot_diags[r - c][v] == 0: del bot_diags[r - c][v]
            ans[r][c] = abs(len(bot_diags[r - c]) - len(top_diags[r - c]))
            top_diags[r - c][v] += 1
        return ans
```

## 2712. Minimum Cost to Make All Characters Equal

### Solution 1:  prefix and suffix array of difference points + find min(prefix[i] + suffix[i]) when swapping to 1s or 0s

```py
class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        # CONSTRUCT PREFIX AND SUFFIX ARRAY OF DIFFERENCE POINTS
        parr = []
        for i in range(1, n):
            if s[i] != s[i - 1]:
                parr.append(i - 1)
        parr.append(n - 1)
        sarr = []
        for i in range(n - 2, -1, -1):
            if s[i] != s[i + 1]:
                sarr.append(i + 1)
        sarr.append(0)
        sarr = sarr[::-1]
        def prefix(ch):
            dp = [0] * (len(parr) + 1)
            dp[1] = parr[0] + 1 if s[parr[0]] == ch else 0
            for i in range(1, len(parr)):
                idx = parr[i]
                if s[idx] == ch:
                    dp[i + 1] = dp[i] + idx + 1 + parr[i - 1] + 1
                else:
                    dp[i + 1] = dp[i]
            return dp
        def suffix(ch):
            dp = [0]*(len(sarr) + 1)
            dp[-2] = n - sarr[-1] if s[sarr[-1]] == ch else 0
            for i in range(len(sarr) - 2, -1, -1):
                idx = sarr[i]
                if s[idx] == ch:
                    dp[i] = dp[i + 1] + (n - idx) + (n - sarr[i + 1])
                else:
                    dp[i] = dp[i + 1]
            return dp
        pref_cost, suf_cost = prefix('1'), suffix('1') # invert 1s to 0s
        res = math.inf
        for i in range(len(parr)):
            res = min(res, pref_cost[i] + suf_cost[i])
        pref_cost, suf_cost = prefix('0'), suffix('0') # invert 0s to 1s
        for i in range(len(parr)):
            res = min(res, pref_cost[i] + suf_cost[i])
        return res
```

### Solution 2:  observation

If you draw it out can find this pattern, but still haven't proved it. 

```py
class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        return sum(mi-n(i, n - i) for i in range(1, n) if s[i] != s[i - 1])
```

## 2713. Maximum Strictly Increasing Cells in a Matrix

### Solution 1:  dynamic programming + start with largest value and work way backwards and take size for each row and column + sort coordinates by value

```py
class Solution:
    def maxIncreasingCells(self, mat: List[List[int]]) -> int:
        R, C = len(mat), len(mat[0])
        row_size, col_size = [0]*R, [0]*C
        prev_row_size, prev_col_size = [0]*R, [0]*C
        prev_row, prev_col = [-math.inf]*R, [-math.inf]*C
        prev_prev_row, prev_prev_col = [-math.inf]*R, [-math.inf]*C
        coords = sorted([(r, c) for r, c in product(range(R), range(C))], key = lambda x: mat[x[0]][x[1]], reverse = True)
        for r, c in coords:
            v = mat[r][c]
            rsize, csize = row_size[r], col_size[c]
            if v == prev_row[r]:
                if prev_prev_row[r] != -math.inf:
                    rsize = prev_row_size[r]
                else:
                    rsize = 0
            if v == prev_col[c]:
                if prev_prev_col[c] != -math.inf:
                    csize = prev_col_size[c]
                else:
                    csize = 0
            size = max(rsize, csize) + 1
            if v != prev_row[r]:
                prev_row_size[r] = row_size[r]
                prev_prev_row[r] = prev_row[r]
                prev_row[r] = v
            if v != prev_col[c]:
                prev_col_size[c] = col_size[c]
                prev_prev_col[c] = prev_col[c]
                prev_col[c] = v
            row_size[r] = max(row_size[r], size)
            col_size[c] = max(col_size[c], size)
        return max(max(row_size), max(col_size))
```


# Leetcode Weekly Contest 348

## 2716. Minimize String Length

### Solution 1:  string + logic

```py
class Solution:
    def minimizedStringLength(self, s: str) -> int:
        return len(set(s))
```

## 2717. Semi-Ordered Permutation

### Solution 1:  math

the number of adjacent swaps to get the integer 1 to the 0 index is basically the index it is currently at. For the integer n, it is how far away it is from the last integer.  But if 1 is to the right of n, then one of the swaps will be to move 1 to the left or n.  In that case subtract 1. 

```py
class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        first, last = nums.index(1), n - nums.index(n) - 1
        return first + last - (1 if nums.index(n) < first else 0)
```

## 2718. Sum of Matrix After Queries

### Solution 1:  matrix + track how many rows and columns have been queried

For this problem, if you go through the queries in reverse then you can compute it in O(n) time.  Because each time you use a row or column, you fill it in with an integer, and never again can you fill in that row that will contribute to the final result. So mark as visited. But also track how many unique rows and columns have been filled with value.  This way when you fill rows, you need to subtract how many columns have been filled up to that point to compute the actual value can get in current query.

```py
class Solution:
    def matrixSumQueries(self, n: int, queries: List[List[int]]) -> int:
        # type, index, value
        res = row_count = col_count = 0
        rows, cols = [0] * n, [0] * n
        for t, i, v in reversed(queries):
            if t == 0:
                if rows[i]: continue
                res += n * v - col_count * v
                row_count += 1
                rows[i] = 1
            else:
                if cols[i]: continue
                res += n * v - row_count * v
                col_count += 1
                cols[i] = 1
        return res
```

## 2719. Count of Integers

### Solution 1:  digit dp

dp(i, j, t) = number of integers with i digits, sum of digits is j, t represents tight bound

```py
class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        mod = 10**9 + 7
        def f(num):
            digits = str(num)
            n = len(digits)
            # dp(i, j, t) ith index in digits, j sum of digits, t represents tight bound
            dp = [[[0] * 2 for _ in range(max_sum + 1)] for _ in range(n + 1)]
            for i in range(min(int(digits[0]), max_sum) + 1):
                dp[1][i][1 if i == int(digits[0]) else 0] += 1
            for i, t, j in product(range(1, n), range(2), range(max_sum + 1)):
                for k in range(10):
                    if t and k > int(digits[i]): break
                    if j + k > max_sum: break
                    dp[i + 1][j + k][t and k == int(digits[i])] += dp[i][j][t]
            return sum(dp[n][j][t] for j, t in product(range(min_sum, max_sum + 1), range(2))) % mod
        num1, num2 = int(num1), int(num2)
        return (f(num2) - f(num1 - 1) + mod) % mod
```

```py
class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        mod = int(1e9) + 7
        def solve(upper):
            dp = Counter({(0, 1): 1})
            for d in map(int, upper):
                ndp = Counter()
                for (dig_sum, tight), cnt in dp.items():
                    for dig in range(10 if not tight else d + 1):
                        ndig_sum = dig_sum + dig
                        if ndig_sum > max_sum: break
                        ntight = tight and dig == d
                        ndp[(ndig_sum, ntight)] = (ndp[(ndig_sum, ntight)] + cnt) % mod
                dp = ndp
            return sum(dp[(ds, t)] for ds, t in product(range(min_sum, max_sum + 1), range(2))) % mod
        return (solve(num2) - solve(str(int(num1) - 1))) % mod
```




# Leetcode Weekly Contest 349

## 2733. Neither Minimum nor Maximum

### Solution 1:  min and max + scan

```py
class Solution:
    def findNonMinOrMax(self, nums: List[int]) -> int:
        min_, max_ = min(nums), max(nums)
        for num in nums:
            if num not in (min_, max_): return num
        return -1
```

## 2734. Lexicographically Smallest String After Substring Operation

### Solution 1:  greedy + decrease the earliest infix that is not containing a

```py
class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        start = n - 1
        for i in range(n):
            if s[i] != 'a':
                start = i
                break
        res = list(s[:start])
        for i in range(start, n):
            if i != start and s[i] == 'a': 
                res.extend(s[i:])
                break
            if s[i] == 'a':
                res.append('z')
                continue
            res.append(chr(ord(s[i]) - 1))
        return ''.join(res)
```

## 2735. Collecting Chocolates

### Solution 1:  simulation

```py
class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        res = sum(nums)
        mins = nums[:]
        for i in range(1, n):
            savings = 0
            v = []
            for j in range(n):
                if nums[(j + i) % n] < mins[j]:
                    savings += (mins[j] - nums[(j + i) % n])
                    v.append((j, nums[(j + i) % n]))
            savings -= x
            if savings <= 0: break
            if savings > 0:
                res -= savings
                for j, val in v:
                    mins[j] = val
        return res
```

## 2736. Maximum Sum Queries

### Solution 1:  two heaps + offline query + sort

```py
class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n, m = len(nums1), len(queries)
        ans = [-1] * m
        queries = sorted([(x, y, i) for i, (x, y) in enumerate(queries)], reverse = True)
        max_heap = []
        heap = []
        nums = sorted([(x1, x2) for x1, x2 in zip(nums1, nums2)], reverse = True)
        j = 0
        for x, y, i in queries:
            while j < n and nums[j][0] >= x:
                if nums[j][1] < y:
                    heappush(heap, (-nums[j][1], -sum(nums[j])))
                else:
                    heappush(max_heap, (-sum(nums[j]), nums[j][1]))
                j += 1
            while heap and abs(heap[0][0]) >= y:
                r, v = heappop(heap)
                heappush(max_heap, (v, abs(r)))
            while max_heap and max_heap[0][1] < y:
                v1, v2 = heappop(max_heap)
                heappush(heap, (-v2, v1))
            if max_heap: ans[i] = abs(max_heap[0][0]) 
        return ans
```

### Solution 2:  maximum segment tree + offline queries + sort + coordinate compression

```py
class SegmentTree:
    def __init__(self, n: int, neutral: int, func):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)]

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] = self.func(self.nodes[segment_idx], val)
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.func(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}, next array: {self.nodes}"

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        queries = sorted([(left, right, i) for i, (left, right) in enumerate(queries)], reverse = True)
        nums = sorted([(n1, n2) for n1, n2 in zip(nums1, nums2)], reverse = True)
        values = set()
        for _, v, _ in queries:
            values.add(v)
        for _, v in nums:
            values.add(v)
        compressed = {}
        for i, v in enumerate(sorted(values)):
            compressed[v] = i
        max_seg_tree = SegmentTree(len(compressed), -1, max)
        ans = [-1] * len(queries)
        i = 0
        for left, right, idx in queries:
            while i < n and nums[i][0] >= left:
                max_seg_tree.update(compressed[nums[i][1]], sum(nums[i]))
                i += 1
            ans[idx] = max_seg_tree.query(compressed[right], len(compressed))
        return ans
```




# Leetcode Weekly Contest 350

## 2739. Total Distance Traveled

### Solution 1: math + O(1)

Assume transfer x liters addtional tank to main tank
1. x <= additionaTank
1. 5x < x + mainTank
solve for second condition leads to x < mainTank / 4
If do a little math can rewrite it as x <= (mainTank - 1) // 4

```py
class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        return (mainTank + min(additionalTank, (mainTank - 1) // 4)) * 10
```

## 2740. Find the Value of the Partition

### Solution 1:  sort + min difference between nearest elements in array

```py
class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        return min(nums[i] - nums[i - 1] for i in range(1, n))
```

## 2741. Special Permutations

### Solution 1:  bitmask dp

dp[mask, last_index] = count of ways to achieve this state

```py
class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        mod = int(1e9) + 7
        dp = {(1 << i, i): 1 for i in range(n)}
        for _ in range(n - 1):
            ndp = Counter()
            for (mask, j), v in dp.items():
                for i in range(n):
                    if (mask >> i) & 1: continue
                    if nums[i] % nums[j] != 0 and nums[j] % nums[i] != 0: continue
                    nstate = (mask | (1 << i), i)
                    ndp[nstate] = (ndp[nstate] + v) % mod
            dp = ndp
        return sum(dp.values()) % mod
```

## 2742. Painting the Walls

### Solution 1:  knapsack dynamic programming

solve for total time and you compute the min for that current subset from 0...i for the current house. 

```py
class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        dp = {0: 0}
        for x, y in zip(cost, time):
            ndp = defaultdict(lambda: math.inf)
            for k, v in dp.items():
                # skip paid painter
                ndp[k - 1] = min(ndp[k - 1], v)
                # assign paid painter
                ndp[k + y] = min(ndp[k + y], v + x)
            dp = ndp
        return min(v for k, v in dp.items() if k >= 0)
```

### Solution 2:  iterative dp + space optimized + array

dp[i] = the min cost to buy enough time to paint i houses. 
so the goal is to calculate the min cost to buy enough time to paint n houses or dp[n]

Each time you choose to buy time to paint a house, you get free painter for time[i] time.
So actually you are buying time[i] + 1 houses painted when spending cost[i].  
The paid painter paints one house, and the free painter can paint time[i] houses. 

```py
class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [0] + [math.inf] * n
        for x, y in zip(cost, time):
            for i in range(n, 0, -1):
                dp[i] = min(dp[i], dp[max(i - y - 1, 0)] + x)
        return dp[-1]
```

# Leetcode Weekly Contest 351

## 2748. Number of Beautiful Pairs

### Solution 1:  gcd + nested loops

```py
class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        n = len(nums)
        get = lambda i, loc: int(str(nums[i])[loc])
        return sum(1 for i in range(n) for j in range(i + 1, n) if math.gcd(get(i, 0), get(j, -1)) == 1)
```

## 2749. Minimum Operations to Make the Integer Zero

### Solution 1:  bit manipulation + math

determine if can form by sum of powers of two with i terms. 

So the lower bound is you need at least the number of 1s in the binary representation of v.  Right cause if you have 1010, you need at least 2 power 2, 2^3 + 2^1, 
The upper bound is you need at least v power of twos, cause you can just do 2^0+2^0 which is same as 1 + 1 + 1, so you can do that in v terms.

But what is tricky is why can you do anything in between, and the reason is that you can always split something into two, that is 2^3 = 2^2+2^2, and in this way you can get to 2^0+2^0+...+2^0, by increasing number of power of twos that sum to 10 all the way until you reach upper bound, by incrementing by one each time, by breaking a 2^x into 2^(x - 1) + 2^(x - 1)

```py
class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for i in range(61):
            v = num1 - i * num2
            if v.bit_count() <= i <= v: return i
        return -1
```

## 2750. Ways to Split Array Into Good Subarrays

### Solution 1:  combinatorics + counting

given some array
0011001001, you can partition any sandwiched 0 segment, which means it lies between 1s. 
And the possible ways to split it is the number of 0s + 1, 
cause for instance
0011001001, you can split that first 0 segment these ways
0011|001001
00110|01001
001100|1001

multiply all the number of partition for each zero segment

```py
class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        n = len(nums)
        if sum(nums) == 0: return 0
        for left in range(n):
            if nums[left] == 1: break
        for right in reversed(range(n)):
            if nums[right] == 1: break
        nums = nums[left: right + 1]
        res = 1
        mod = 10 ** 9 + 7
        for key, grp in groupby(nums):
            if key == 0:
                res *= len(list(grp)) + 1
                res %= mod
        return res
```

## 2751. Robot Collisions

### Solution 1:  stack + greedy

```py
class Solution:
    def survivedRobotsHealths(self, positions: List[int], healths: List[int], directions: str) -> List[int]:
        n = len(positions)
        robots = sorted(range(n), key = lambda i: positions[i])
        stack = []
        for i in robots:
            if directions[i] == 'L':
                while stack and directions[stack[-1]] == 'R' and healths[i] > 0:
                    idx = stack.pop()
                    if healths[idx] < healths[i]:
                        healths[i] -= 1
                        healths[idx] = 0
                    elif healths[idx] == healths[i]:
                        healths[i] = healths[idx] = 0
                    elif healths[idx] > healths[i]:
                        healths[i] = 0
                        healths[idx] -= 1
                    if healths[idx] > 0:
                        stack.append(idx)
            if healths[i] > 0:
                stack.append(i)
        return filter(lambda x: x > 0, healths)
```

# Leetcode Weekly Contest 352

## 2760. Longest Even Odd Subarray With Threshold

### Solution 1:  sliding window

```py
class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        left = res = 0
        while left < n:
            while left < n and nums[left] & 1:
                left += 1
            if left < n and nums[left] > threshold:
                left += 1
                continue
            right = left
            while right + 1 < n and nums[right] % 2 != nums[right + 1] % 2 and nums[right + 1] <= threshold:
                right += 1
            if left < n:
                res = max(res, right - left + 1)
            left = right + 1
        return res
```

## 2761. Prime Pairs With Target Sum

### Solution 1:  prime sieve

```py
def prime_sieve(lim):
    primes = [1] * lim
    primes[0] = primes[1] = 0
    p = 2
    while p * p <= lim:
        if primes[p]:
            for i in range(p * p, lim, p):
                primes[i] = 0
        p += 1
    return primes

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        primes = prime_sieve(n + 1)
        res = []
        for x in range(2, n):
            y = n - x
            if x > y: break
            if primes[x] and primes[y]:
                res.append([x, y])
        return res
```

## 2762. Continuous Subarrays

### Solution 1: sliding window + monotonic deque

```py
class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        res = left = 0
        min_stack, max_stack = deque(), deque()
        for right in range(n):
            while min_stack and nums[right] <= nums[min_stack[-1]]:
                min_stack.pop()
            while max_stack and nums[right] >= nums[max_stack[-1]]:
                max_stack.pop()
            min_stack.append(right)
            max_stack.append(right)
            while abs(nums[max_stack[0]] - nums[min_stack[0]]) > 2:
                left += 1
                while min_stack and min_stack[0] < left:
                    min_stack.popleft()
                while max_stack and max_stack[0] < left:
                    max_stack.popleft()
            delta = right - left + 1
            res += delta
        return res
```

## 2763. Sum of Imbalance Numbers of All Subarrays

### Solution 1:  hash table + track imbalance based on conditions

```py
class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        res = 0
        n = len(nums)
        for i in range(n):
            seen = [0] * (n + 2)
            seen[nums[i]] = 1
            min_val = max_val = nums[i]
            imbalance = 0
            for j in range(i + 1, n):
                prv, val, nxt = nums[j] - 1, nums[j], nums[j] + 1
                if nums[j] > max_val and not seen[prv]: imbalance += 1
                if nums[j] < min_val and not seen[nxt]: imbalance += 1
                if min_val < val < max_val and seen[prv] and seen[nxt] and not seen[val]: imbalance -= 1
                if min_val < val < max_val and not seen[prv] and not seen[nxt] and not seen[val]: imbalance += 1
                min_val = min(min_val, nums[j])
                max_val = max(max_val, nums[j])
                seen[val] = 1
                res += imbalance
        return res
```

# Leetcode Weekly Contest 353

## 2769. Find the Maximum Achievable Number

### Solution 1:  math

```py
class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        return num + 2 * t
```

## 2770. Maximum Number of Jumps to Reach the Last Index

### Solution 1:  dynamic programming + O(n^2)

dp[i] = the maximum number of jumps to get to the nums[j]

```py
class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [0] + [-math.inf] * (n - 1)
        for j in range(1, n):
            for i in range(j):
                if abs(nums[i] - nums[j]) <= target:
                    dp[j] = max(dp[j], dp[i] + 1)
        return dp[-1] if dp[-1] != -math.inf else -1
```

## 2771. Longest Non-decreasing Subarray From Two Arrays

### Solution 1:  dynamic programming + space optimized

dp1[i] is maximum longest non-decreasing substring ending with nums1[i]
dp2[i] is maximum longest non-decreasing substring ending with nums2[i]

```py
class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        dp1 = dp2 = res = 1
        for i in range(1, n1):
            ndp1 = ndp2 = 1
            if nums1[i] >= nums1[i - 1]:
                ndp1 = max(ndp1, dp1 + 1)
            if nums1[i] >= nums2[i - 1]:
                ndp1 = max(ndp1, dp2 + 1)
            if nums2[i] >= nums2[i - 1]:
                ndp2 = max(ndp2, dp2 + 1)
            if nums2[i] >= nums1[i - 1]:
                ndp2 = max(ndp2, dp1 + 1)
            dp1, dp2 = ndp1, ndp2
            res = max(res, dp1, dp2)
        return res
```

## 2772. Apply Operations to Make All Array Elements Equal to Zero

### Solution 1:  difference array + construct array from all 0s + backwards

```py
class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        cur = 0
        for i in range(n):
            cur += diff[i] # update the changes
            delta = nums[i] - cur
            if delta < 0: return False
            if delta > 0 and i + k > n: return False
            if delta > 0:
                cur += delta
                diff[i + k] -= delta
        return True
```

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

# Leetcode Weekly Contest 355

## 2788. Split Strings by Separator

### Solution 1: 

```py
class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        res = []
        for word in words:
            res.extend(filter(None, word.split(separator)))
        return res
```

## 2789. Largest Element in an Array after Merge Operations

### Solution 1: 

```py
class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        res = prv = 0
        for num in reversed(nums):
            if num <= prv:
                prv += num
            else:
                prv = num
            res = max(res, prv)
        return res
```

## 2790. Maximum Number of Groups With Increasing Length

### Solution 1:  sort + greedy

![images](images/number_groups_with_increasing_length.png)

```py
class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        cur = res = 0
        for usage in sorted(usageLimits):
            cur += usage
            if cur > res:
                res += 1
                cur -= res
        return res
```

## 2791. Count Paths That Can Form a Palindrome in a Tree

### Solution 1:  dynamic programming + modulus 2 addition + xor + lowest common ancestor

![image](images/count_number_palindromes_1.png)
![image](images/count_number_palindromes_2.png)
![image](images/count_number_palindromes_3.png)

```py
class Solution:
    def countPalindromePaths(self, parent: List[int], s: str) -> int:
        n = len(parent)
        @cache
        def mask(node):
            i = ord(s[node]) - ord('a')
            return mask(parent[node]) ^ (1 << i) if node else 0
        count = Counter()
        res = 0
        for i in range(n):
            cur_mask = mask(i)
            res += count[cur_mask] + sum(count[cur_mask^ (1 << j)] for j in range(26))
            count[cur_mask] += 1
        return res
```

# Leetcode Weekly Contest 356

## 2798. Number of Employees Who Met the Target

### Solution 1:  sum

```py
class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        return sum(1 for h in hours if h >= target)
```

## 2799. Count Complete Subarrays in an Array

### Solution 1:  set + brute force

```py
class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        t = len(set(nums))
        res = 0
        for i in range(n):
            cur = set()
            for j in range(i, n):
                cur.add(nums[j])
                if len(cur) == t: res += 1
        return res
```

## 2800. Shortest String That Contains Three Strings

### Solution 1:  try all permutations

```py

```

## 2801. Count Stepping Numbers in Range

### Solution 1:  digit dp

three tight values, because sometimes you can have a prefix that is larger than current, or smaller or it could be equal.  
0 = smaller, 1 = equal, 2 = larger
Then just need to use correct argument to update the count for states, but also need to track how many of these integers that are smaller than digits which contributes to the number of stepping integers.

For instance if it is a tight bound, and it is smaller than you can move to the smaller
if it is a tight bound, and it is larger than you can move to the larger
if it is a tight bound, and it is equal than you stay with tight bound

if it is smaller than you can add any valid digit that satisfies the constraint.
if it is larger you can add it unless it is for the last digit, then you can't add it because it will be larger than the upper bound.

```py
class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = int(1e9) + 7
        # (last_dig, tight, zero)
        def solve(upper):
            dp = Counter({(0, 1, 1): 1})
            for d in map(int, upper):
                ndp = Counter()
                for (last_dig, tight, zero), cnt in dp.items():
                    for dig in range(10 if not tight else d + 1):
                        if not zero and abs(last_dig - dig) != 1: continue
                        ntight, nzero = tight and dig == d, zero and dig == 0
                        ndp[(dig, ntight, nzero)] = (ndp[(dig, ntight, nzero)] + cnt) % mod
                dp = ndp
            return sum(dp[(dig, t, 0)] for dig, t in product(range(10), range(2))) % mod
        low_is_stepping_int = all(abs(x - y) == 1 for x, y in zip(map(int ,low), map(int, low[1:])))
        return (solve(high) - solve(low) + low_is_stepping_int) % mod
```

# Leetcode Weekly Contest 357

## 2810. Faulty Keyboard

### Solution 1:  deque

```py
class Solution:
    def finalString(self, s: str) -> str:
        n = len(s)
        res = []
        for ch in s:
            if ch == 'i':
                res.reverse()
            else:
                res.append(ch)
        return ''.join(res)
```

### Solution 2:  flipping + deque + O(n) + forward facing and reverse facing

```py
class Solution:
    def finalString(self, s: str) -> str:
        queue = deque()
        forward = False
        for chars in s.split('i'):
            forward = not forward
            if forward:
                queue.extend(chars)
            else:
                queue.extendleft(chars)
        return ''.join(queue) if forward else ''.join(reversed(queue))
```

## 2811. Check if it is Possible to Split Array

### Solution 1:  any + greedy

Observe that you can split any subarray into a single element and the rest, so you just need to do that and have one part for when you get to 3 elements in array, so that a part of it is greater than or equal to m.  That way you can remove the one element.

```py
class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        return len(nums) <= 2 or any(nums[i] + nums[i - 1] >= m for i in range(1, len(nums)))
```

## 2812. Find the Safest Path in a Grid

### Solution 1:  bfs + deque + dijkstra + max heap

Set the grid integers to be equal to the minimum distance to a thief, can do this with a multisource bfs from each thief.  Use those grid integers with a max heap.  Can convert the problem to min heap as well.  

```py
class Solution:
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        n = len(grid)
        thief, empty = 1, 0
        queue = deque([(r, c, 0) for r, c in product(range(n), repeat = 2) if grid[r][c] == thief])
        grid = [[-1] * n for _ in range(n)]
        for r, c, _ in queue:
            grid[r][c] = 0
        neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
        while queue:
            r, c, lv = queue.popleft()
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or grid[nr][nc] != -1: continue
                grid[nr][nc] = lv + 1
                queue.append((nr, nc, lv + 1))
        maxheap = [(-grid[0][0], 0, 0)]
        grid[0][0] = -1
        while maxheap:
            lv, r, c = heappop(maxheap)
            lv = abs(lv)
            if r == c == n - 1:
                return lv
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or grid[nr][nc] == -1: continue
                grid[nr][nc] = min(lv, grid[nr][nc])
                heappush(maxheap, (-grid[nr][nc], nr, nc))
                grid[nr][nc] = -1
        return 0
```

### Solution 2:  bfs + binary search + bisect_left + dfs

```py
class Solution:
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        n = len(grid)
        thief, empty = 1, 0
        queue = deque()
        for r, c in product(range(n), repeat = 2):
            if grid[r][c] == thief:
                grid[r][c] = 0
                queue.append((r, c, 0))
            else:
                grid[r][c] = -1
        neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
        while queue:
            r, c, lv = queue.popleft()
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or grid[nr][nc] != -1: continue
                grid[nr][nc] = lv + 1
                queue.append((nr, nc, lv + 1))
        def possible(target):
            stack = []
            vis = set()
            if grid[0][0] >= target:
                stack.append((0, 0))
                vis.add((0, 0))
            while stack:
                r, c = stack.pop()
                if r == c == n - 1: return False
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or grid[nr][nc] < target or (nr, nc) in vis: continue
                    stack.append((nr, nc))
                    vis.add((nr, nc))
            return True
        # FFFTTT, last F
        # false for when you can reach end with the safeness factor
        # true if you can't reach
        return bisect_right(range(2 * n + 1), False, key = lambda x: possible(x)) - 1
```

## 2813. Maximum Elegance of a K-Length Subsequence

### Solution 1:  greedy + sort + monotonic stack

```py
class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        n = len(items)
        seen = set()
        res = cur = 0
        # extra will be sorted in non-increasing profits, so the lowest profit at the end of it
        # this will be useful for when you want to add a new category to the list, and you can do this
        # by replacing it. 
        extra = []
        for i, (profit, category) in enumerate(sorted(items, reverse = True)):
            if i < k:
                cur += profit
                if category in seen:
                    extra.append(profit)
            elif category not in seen:
                if not extra: break
                cur += profit - extra.pop()
            seen.add(category)
            res = max(res, cur + len(seen) * len(seen))
        return res
```

### Solution 2: min and max heaps + dictionary

```py
class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        n = len(items)
        max_item = Counter()
        index = dict()
        vis = set()
        categories = [0] * (n + 1)
        for i, (profit, cat) in enumerate(items):
            if profit > max_item[cat]:
                max_item[cat] = profit
                index[cat] = i
        res = cur = 0
        min_heap = []
        num_cat = 0
        for cat, pr in sorted(max_item.items(), key = lambda pair: (-pair[1])):
            cur += pr
            vis.add(index[cat])
            categories[cat] = 1
            num_cat += 1
            k -= 1
            heappush(min_heap, (pr, cat))
            if k == 0: break
        res = cur + num_cat * num_cat
        # only add items to the max heap which are not visited
        remain_items = sorted([(profit, cat) for i, (profit, cat) in enumerate(items) if i not in vis], reverse = True)
        for profit, cat in remain_items:
            if k == 0:
                k += 1
                pr, ca = heappop(min_heap)
                cur -= pr
                categories[ca] -= 1
                if categories[ca] == 0:
                    num_cat -= 1
            k -= 1
            cur += profit
            categories[cat] += 1
            if categories[cat] == 1:
                num_cat += 1
            heappush(min_heap, (profit, cat))
            res = max(res, cur + num_cat * num_cat)
        return res
```

# Leetcode Weekly Contest 358

## 2815. Max Pair Sum in an Array

### Solution 1:  defaultdict + sort 

the hashed value for each list is the maximum digit in that num, so they are all grouped together, and then just take the two largest from each group.

```py
class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        d = defaultdict(list)
        for num in nums:
            dig = max(map(int, str(num)))
            d[dig].append(num)
        res = -1
        for vals in d.values():
            if len(vals) == 1: continue
            vals.sort(reverse=True)
            res = max(res, vals[0] + vals[1])
        return res
```

## 2816. Double a Number Represented as a Linked List

### Solution 1:  linked lists + reverse

```py
class Solution:
    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        digs = []
        cur = head
        while cur:
            digs.append(cur.val)
            cur = cur.next
        carry = 0
        digs.reverse()
        res = []
        for dig in digs:
            cur = dig * 2 + carry
            res.append(cur % 10)
            carry = cur // 10
        if carry:
            res.append(carry)
        head = ListNode(res.pop())
        cur = head
        while res:
            cur.next = ListNode(res.pop())
            cur = cur.next
        return head
```

## 2817. Minimum Absolute Difference Between Elements With Constraint

### Solution 1:  sortedlist + binary search

```py
from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        sl = SortedList()
        res = math.inf
        for i in range(x, n):
            sl.add(nums[i-x])
            j = sl.bisect_right(nums[i])
            if j < len(sl):
                res = min(res, abs(nums[i] - sl[j]))
            if j > 0:
                res = min(res, abs(nums[i] - sl[j - 1]))
        return res
```

## 2818. Apply Operations to Maximize Score

### Solution 1:  monotonic stack + prime factorization + prime count + sort + offline queries + pow

prime_count function counts the number of prime factors for any given num. 



```py
def prime_count(num):
    cnt = 0
    i = 2
    while i * i <= num:
        cnt += num % i == 0
        while num % i == 0:
            num //= i
        i += 1
    cnt += num > 1
    return cnt

class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        mod = int(1e9) + 7
        n = len(nums)
        pscores = list(map(lambda x: prime_count(x), nums))
        left, right = [0] * n, [0] * n
        # forward pass to compute the greater right elements
        # find how much forward from an element it can go and be the largest
        stk = []
        for i, p in enumerate(pscores + [math.inf]):
            while stk and pscores[stk[-1]] < p:
                j = stk.pop()
                right[j] = i - 1
            stk.append(i)
        # backward pass to compute the lesser left elements
        # find howmuch back from an element it can go and be the largest
        stk = []
        for i, p in zip(range(n - 1, -2, -1), reversed([math.inf] + pscores)):
            if stk:
                index = stk[-1]
            while stk and pscores[stk[-1]] <= p:
                j = stk.pop()
                left[j] = index
            stk.append(i)
        queries = sorted([(num, i) for i, num in enumerate(nums)], reverse = True)
        res = 1
        for num, i in queries:
            left_, right_ = i - left[i] + 1, right[i] - i + 1
            t = min(left_ * right_, k)
            res = (res * pow(num, t, mod)) % mod
            k -= t
            if k == 0: break
        return res
```



# Leetcode Weekly Contest 359

## 2828. Check if a String Is an Acronym of Words

### Solution 1:  all + zip_longest + loop

```py
class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        return all(c1 == c2 for c1, c2 in zip_longest(map(lambda x: x[0], words), s, fillvalue = "#"))
```

## 2829. Determine the Minimum Sum of a k-avoiding Array

### Solution 1:  set + greedy

add the smallest numbers first

```py
class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        arr = set()
        i = 1
        while len(arr) != n:
            if k - i not in arr:
                arr.add(i)
            i += 1
        return sum(arr)
```

## 2830. Maximize the Profit as the Salesman

### Solution 1:  dynamic programming + interval + O(n)

given a start, end, gold, 
you already calculated the maximum up to the start house since start is smaller than end
So just take the value from before start, so start-1 and add gold to that

```py
class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        A = defaultdict(list)
        for start, end, gold in offers:
            A[end + 1].append((start + 1, gold))
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1]
            for start, gold in A[i]:
                dp[i] = max(dp[i], dp[start - 1] + gold)
        return dp[-1]
            
```

## 2831. Find the Longest Equal Subarray

### Solution 1:  sliding window over each num + sort indices for each num

So this is like a bucketized sliding window algorithm. Each number can be viewed as a bucket and an independent sliding window algorithm runs on each bucket. 
Within each bucket you need to have the indices sorted for where that num is in nums array. This allows you to compute the number that you delete over any interval and also know the size of the num you can get with that number of deletions. 

suppose num = 2
and indices = [1,4,6,9]
so 
1xx1x1xx1, where x can be any other number 
left = 0
and right = 3
These are the pointers for the indices array
that means you are currently considering right - left + 1 = 4 elements of 1, but you need to delete some to get them to be adjacent to each other
you can count the x above which is 5 deletions. 
and you can know it because the length of from the indices[right] - indices[left] + 1 is the size of the entire subarray in consideration.  so it is 9 here, so you just take the 10 and subtract the 4 to get 5.  

So you just need to maximize the right - left + 1, for when it doesn't exceed the k deletions. 

```py
class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        indices = [[] for _ in range(n + 1)]
        for i, num in enumerate(nums):
            indices[num].append(i)
        res = 0
        for index in indices:
            if not index: continue
            left = 0
            for right in range(len(index)):
                while (index[right] - index[left]) - (right - left) > k: left += 1
                res = max(res, right - left + 1)
        return res
```



# Leetcode Weekly Contest 361

## 2843. Count Symmetric Integers

### Solution 1: 

```py
class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        res = 0
        for i in range(low, high + 1):
            digits = str(i)
            n = len(digits)
            if n & 1: continue
            sum1, sum2 = sum(int(digits[j]) for j in range(n // 2)), sum(int(digits[j]) for j in range(n // 2, n))
            res += sum1 == sum2
        return res
```

## 2844. Minimum Operations to Make a Special Number

### Solution 1: 

```py
class Solution:
    def minimumOperations(self, num: str) -> int:
        stack = list(num)
        match = {"25", "50", "75", "00"}
        last = set()
        n = len(num)
        for i in reversed(range(n)):
            if num[i] in "27" and "5" in last:
                return n - i - 2
            if num[i] in "50" and "0" in last:
                return n - i - 2
            last.add(num[i])
        return n - ("0" in num)
```

## 2845. Count of Interesting Subarrays

### Solution 1:  dp + math

```py

```

## 2846. Minimum Edge Weight Equilibrium Queries in a Tree

### Solution 1:  binary jumping + lowest common ancestor (LCA) + frequency array for each weight + tree

The minimum operations is by not changing the one with highest frequency and thus would require most operations. 

```py
class Solution:
    def minOperationsQueries(self, n: int, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
        adj_list = [[] for _ in range(n)]
        for u, v, w in edges:
            adj_list[u].append((v, w))
            adj_list[v].append((u, w))
        LOG = 14
        depth = [0] * n
        parent = [-1] * n
        freq = [[0] * 27 for _ in range(n)]
        # CONSTRUCT THE PARENT, DEPTH AND FREQUENCY ARRAY FROM ROOT
        def dfs(root):
            queue = deque([root])
            vis = [0] * n
            vis[root] = 1
            dep = 0
            while queue:
                for _ in range(len(queue)):
                    node = queue.popleft()
                    depth[node] = dep
                    for nei, wei in adj_list[node]:
                        if vis[nei]: continue
                        freq[nei] = freq[node][:]
                        freq[nei][wei] += 1
                        parent[nei] = node
                        vis[nei] = 1
                        queue.append(nei)
                dep += 1
        dfs(0)
        # CONSTRUCT THE SPARSE TABLE FOR THE BINARY JUMPING TO ANCESTORS IN TREE
        ancestor = [[-1] * n for _ in range(LOG)]
        ancestor[0] = parent[:]
        for i in range(1, LOG):
            for j in range(n):
                if ancestor[i - 1][j] == -1: continue
                ancestor[i][j] = ancestor[i - 1][ancestor[i - 1][j]]
        def kth_ancestor(node, k):
            for i in range(LOG):
                if (k >> i) & 1:
                    node = ancestor[i][node]
            return node
        def lca(u, v):
            # ASSUME NODE u IS DEEPER THAN NODE v   
            if depth[u] < depth[v]:
                u, v = v, u
            # PUT ON SAME DEPTH BY FINDING THE KTH ANCESTOR
            k = depth[u] - depth[v]
            u = kth_ancestor(u, k)
            if u == v: return u
            for i in reversed(range(LOG)):
                if ancestor[i][u] != ancestor[i][v]:
                    u, v = ancestor[i][u], ancestor[i][v]
            return ancestor[0][u]
        ans = [None] * len(queries)
        for i, (u, v) in enumerate(queries):
            lca_node = lca(u, v)
            freqs = [freq[u][w] + freq[v][w] - 2 * freq[lca_node][w] for w in range(27)]
            ans[i] = sum(freqs) - max(freqs)
        return ans
```



# Leetcode Weekly Contest 362

## 2848. Points That Intersect With Cars

### Solution 1: 

```py
class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        vis = [0] * 101
        for s, e in nums:
            for i in range(s, e + 1):
                vis[i] = 1
        return sum(vis)
```

## 2849. Determine if a Cell Is Reachable at a Given Time

### Solution 1:  math

```py
class Solution:
    def isReachableAtTime(self, sx: int, sy: int, fx: int, fy: int, t: int) -> bool:
        dx, dy = abs(sx - fx), abs(sy - fy)
        x = min(dx, dy) + max(dx - min(dx, dy), dy - min(dx, dy))
        if x == 0:
            return t != 1
        return x <= t
```

## 2850. Minimum Moves to Spread Stones Over Grid

### Solution 1:  grid + backtracking

```py
class Solution:
    def minimumMoves(self, grid: List[List[int]]) -> int:
        n = len(grid)
        res = math.inf
        cur = 0
        manhattan_dist = lambda r1, c1, r2, c2: abs(r1 - r2) + abs(c1 - c2)
        def backtrack(i):
            nonlocal res, cur
            if i == len(cells):
                if all(grid[r][c] == 1 for r, c in product(range(n), repeat = 2)): res = min(res, cur)
                return
            row, col = cells[i]
            for r, c in product(range(n), repeat = 2):
                if (r, c) == (row, col): continue
                if grid[r][c] > 0:
                    dist = manhattan_dist(r, c, row, col)
                    cur += dist
                    grid[r][c] -= 1
                    grid[row][col] += 1
                    backtrack(i + 1)
                    cur -= dist
                    grid[r][c] += 1
                    grid[row][col] -= 1     
        cells = [(r, c) for r, c in product(range(n), repeat = 2) if grid[r][c] == 0]
        backtrack(0)
        return res
```

## 2851. String Transformation

### Solution 1:  matrix exponentiation + kmp + lcp

```py
"""
matrix multiplication with modulus
"""
def mat_mul(mat1, mat2, mod):
    result_matrix = []
    for i in range(len(mat1)):
        result_matrix.append([0]*len(mat2[0]))
        for j in range(len(mat2[0])):
            for k in range(len(mat1[0])):
                result_matrix[i][j] += (mat1[i][k]*mat2[k][j])%mod
    return result_matrix

"""
matrix exponentiation with modulus
matrix is represented as list of lists in python
"""
def mat_pow(matrix, power, mod):
    if power<=0:
        print('n must be non-negative integer')
        return None
    if power==1:
        return matrix
    if power==2:
        return mat_mul(matrix, matrix, mod)
    t1 = mat_pow(matrix, power//2, mod)
    if power%2 == 0:
        return mat_mul(t1, t1, mod)
    return mat_mul(t1, mat_mul(matrix, t1, mod), mod)

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        mod = int(1e9) + 7
        def lcp(pat):
            dp = [0] * n
            j = 0
            for i in range(1, n):
                if pat[i] == pat[j]:
                    j += 1
                    dp[i] = j
                    continue
                while j > 0 and pat[i] != pat[j]:
                    j -= 1
                dp[i] = j
                if pat[i] == pat[j]:
                    j += 1
            return dp
        def kmp(text, pat):
            j = cnt = 0
            for i in range(2 * n - 1):
                while j > 0 and text[i % n] != pat[j]:
                    j = lcp_arr[j - 1]
                if text[i % n] == pat[j]:
                    j += 1
                if j == n:
                    cnt += 1
                    j = lcp_arr[j - 1]
            return cnt
        lcp_arr = lcp(t)
        m = kmp(s, t)
        res = 0
        T = [[max(0, n - m - 1), n - m], [m, max(0, m - 1)]]
        B = [[int(s != t)], [int(s == t)]]
        T_k = mat_pow(T, k, mod)
        M = mat_mul(T_k, B, mod)
        res = M[1][0]
        return res
```



# Leetcode Weekly Contest 363

## 2859. Sum of Values at Indices With K Set Bits

### Solution 1:  loop + bit manipulation

```py
class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        return sum(num for i, num in enumerate(nums) if i.bit_count() == k)
```

## 2860. Happy Students

### Solution 1:  sort + greedy

```py
class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.extend([-math.inf, math.inf])
        nums.sort()
        return sum(1 for i in range(n + 1) if i < nums[i + 1] and i > nums[i])
```

## 2861. Maximum Number of Alloys

### Solution 1: binary search + greedy

```py
class Solution:
    def maxNumberOfAlloys(self, n: int, k: int, budget: int, composition: List[List[int]], stock: List[int], cost: List[int]) -> int:
        def possible(target):
            min_cost = math.inf
            for i in range(k):
                tcost = 0
                for j in range(n):
                    required = composition[i][j] * target
                    if required > stock[j]:
                        tcost += (required - stock[j]) * cost[j]
                min_cost = min(min_cost, tcost)
            return min_cost <= budget
        left, right = 0, 10**16
        while left < right:
            mid = (left + right + 1) >> 1
            if possible(mid):
                left = mid
            else:
                right = mid - 1
        return left
```

## 2862. Maximum Element-Sum of a Complete Subset of Indices

### Solution 1:  prime sieve + math + number theory + hash + counter

The idea is to store all the states which represnt all the prime factors that have an odd power. Then we can use a counter to store the sum of the elements that have the same state. Because anytime they all have same set of prime factors with odd power they can belong to the same set.  

```py
class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        def prime_sieve(upper_bound):
            prime_factorizations = [[] for _ in range(upper_bound + 1)]
            for i in range(2, upper_bound + 1):
                if len(prime_factorizations[i]) > 0: continue # not a prime
                for j in range(i, upper_bound + 1, i):
                    prime_factorizations[j].append(i)
            return prime_factorizations
        pf = prime_sieve(n + 1)
        def search(factors, val):
            p = [0] * len(factors)
            for i, f in enumerate(factors):
                while val % f == 0:
                    p[i] += 1
                    val //= f
            return p
        sums = Counter()
        for i, num in enumerate(nums, start = 1):
            factors = pf[i]
            pcount = search(factors, i)
            state = tuple((f for f, p in zip(factors, pcount) if p & 1))
            sums[state] += num
        return max(sums.values())
```



# Leetcode Weekly Contest 364

## 2864. Maximum Odd Binary Number

### Solution 1: greedy + place 1 at least signficant bit position

```py
class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        n = len(s)
        ones = s.count("1") - 1
        res = [0] * n
        res[-1] = 1
        for i in range(ones):
            res[i] = 1
        return "".join(map(str, res))
```

## 2866. Beautiful Towers II

### Solution 1: prefix and suffix sum + max heaps

Basically consider each index in prefix and suffix to be the peak of the mountain, then the suffix and prefix are the largest sum when the peak is at that index.

```py
class Solution:
    def maximumSumOfHeights(self, H: List[int]) -> int:
        n = len(H)
        heap = []
        res = 0
        psum = [0] * (n + 1)
        for i in range(n):
            cur = lost = 0
            while heap and abs(heap[0][0]) > H[i]:
                v, c = heappop(heap)
                v = abs(v)
                delta = v - H[i]
                lost += delta * c
                cur += c
            heappush(heap, (-H[i], cur + 1))
            psum[i + 1] = psum[i] + H[i] - lost
        heap = []
        ssum = [0] * (n + 1)
        for i in reversed(range(n)):
            cur = lost = 0
            while heap and abs(heap[0][0]) > H[i]:
                v, c = heappop(heap)
                v = abs(v)
                delta = v - H[i]
                lost += delta * c
                cur += c
            heappush(heap, (-H[i], cur + 1))
            ssum[i] = ssum[i + 1] + H[i] - lost
        return max(p + s for p, s in zip(psum, ssum))

```

## 2867. Count Valid Paths in a Tree

### Solution 1: union find + prime sieve

Merge each composite connected component with an adjacent prime number, and increase the size of the neighbor nodes for that prime node, because any other adjacent composite numbers can go through it to have paths with everything attached to it. 

```py
def prime_sieve(lim):
    primes = [1] * lim
    primes[0] = primes[1] = 0
    p = 2
    while p * p <= lim:
        if primes[p]:
            for i in range(p * p, lim, p):
                primes[i] = 0
        p += 1
    return primes

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
    def size_(self, i):
        return self.size[self.find(i)]
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'

class Solution:
    def countPaths(self, n: int, edges: List[List[int]]) -> int:
        ps = prime_sieve(n + 1)
        dsu = UnionFind(n + 1)
        for u, v in edges:
            if ps[u] == ps[v] == 0: dsu.union(u, v) # union composite nodes
        res = 0
        count = [1] * (n + 1)
        for u, v in edges:
            if ps[u] ^ ps[v]: # prime and composite node
                if not ps[u]: u, v = v, u
                res += count[u] * dsu.size_(v)
                count[u] += dsu.size_(v)
        return res
```



# Leetcode Weekly Contest 365

## 2874. Maximum Value of an Ordered Triplet II

### Solution 1:  prefix max

```py
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        pmax = delta = -math.inf
        res = 0
        for num in nums:
            res = max(res, num * delta)
            delta = max(delta, pmax - num)
            pmax = max(pmax, num)
        return res
```

### Solution 2:  prefix max + suffix max

```py
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        pmax = -math.inf
        smax = [0] * (n + 1)
        for i in reversed(range(n)):
            smax[i] = max(smax[i + 1], nums[i])
        res = 0
        for i, num in enumerate(nums):
            res = max(res, (pmax - num) * smax[i + 1])
            pmax = max(pmax, num)
        return res
```

## 2875. Minimum Size Subarray in Infinite Array

### Solution 1:  modular arithmetic + sliding window

```py
class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        sum_ = sum(nums)
        middle_len = (target // sum_) * n
        target %= sum_
        cur = left = 0
        res = math.inf
        for right in range(2 * n):
            cur += nums[right % n]
            while cur > target:
                cur -= nums[left % n]
                left += 1
            if cur == target:
                res = min(res, middle_len + right - left + 1)
        return res if res < math.inf else -1
```

## 2876. Count Visited Nodes in a Directed Graph

### Solution 1:  functional graph + recover path + cycle detection

```py
class Solution:
    def countVisitedNodes(self, edges: List[int]) -> List[int]:
        n = len(edges)
        ans, vis = [0] * n, [0] * n
        def search(u):
            parent = {u: None}
            is_cycle = False
            while True:
                vis[u] = 1
                v = edges[u]
                if v in parent: 
                    is_cycle = True
                    break
                if vis[v]: break
                parent[v] = u
                u = v
            if is_cycle:
                crit_point = parent[edges[u]]
                cycle_path = []
                while u != crit_point:
                    cycle_path.append(u)
                    u = parent[u]
                len_ = len(cycle_path)
                for val in cycle_path:
                    ans[val] = len_
            while u is not None:
                ans[u] = ans[edges[u]] + 1
                u = parent[u]
        for i in range(n):
            if vis[i]: continue
            search(i)
        return ans
```



# Leetcode Weekly Contest 367

## 2904. Shortest and Lexicographically Smallest Beautiful String

### Solution 1:  sliding window

```py
class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        res = ""
        mx = math.inf
        n = len(s)
        wcount = j = 0
        for i in range(n):
            wcount += s[i] == "1"
            while wcount >= k:
                if i - j + 1 < mx:
                    mx = i - j + 1
                    res = s[j : i + 1]
                elif i - j + 1 == mx:
                    res = min(res, s[j : i + 1])
                wcount -= s[j] == "1"
                j += 1
        return res
```

## 2905. Find Indices With Index and Value Difference II

### Solution 1:  offline query, sort, sliding window, min and max

```py
class Solution:
    def findIndices(self, nums: List[int], id: int, vd: int) -> List[int]:
        n = len(nums)
        queries = sorted([(v, i) for i, v in enumerate(nums)])
        window = deque()
        first = math.inf
        last = -math.inf
        for v, i in queries:
            window.append((v, i))
            while window and v - window[0][0] >= vd:
                pv, pi = window.popleft()
                first = min(first, pi)
                last = max(last, pi)
            if i - first >= id: return [first, i]
            if last - i >= id: return [i, last]
        return [-1, -1]
```

## 2906. Construct Product Matrix

### Solution 1:  prefix product, suffix product

```py
class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        R, C = len(grid), len(grid[0])
        mod = 12345
        pprod = 1
        sprod = [1] * (R * C + 1)
        i = -2
        for r in reversed(range(R)):
            for c in reversed(range(C)):
                sprod[i] = (sprod[i + 1] * grid[r][c]) % mod
                i -= 1
        i = 1
        for r, c in product(range(R), range(C)):
            nprod = (pprod * grid[r][c]) % mod
            grid[r][c] = (pprod * sprod[i]) % mod
            pprod = nprod
            i += 1
        return grid
```

# Leetcode Weekly Contest 369

## 2918. Minimum Equal Sum of Two Arrays After Replacing Zeros

### Solution 1:  greedy

```py
class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        n1, n2 = len(nums1), len(nums2)
        c1, c2 = nums1.count(0), nums2.count(0)
        s1, s2 = sum(nums1), sum(nums2)
        if c1 == 0 and c2 == 0 and s1 != s2: return -1
        elif c1 == 0 and s1 < s2 + c2: return -1
        elif c2 == 0 and s2 < s1 + c1: return -1
        return max(s1 + c1, s2 + c2)
```

## 2919. Minimum Increment Operations to Make Array Beautiful

### Solution 1:  dynamic programming + sliding window 

![images](images/Minimum_Increment_Operations_to_Make_Array_Beautiful.png)

```py
class Solution:
    def minIncrementOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        diff = [max(0, k - num) for num in nums]
        dp = diff[:3]
        for i in range(3, n):
            dp[i % 3] = min(dp) + diff[i]
        return min(dp)
```

## 2920. Maximum Points After Collecting Coins From All Nodes

### Solution 1:  depth first search, tree, dynamic programming on a tree

```py
class Solution:
    def maximumPoints(self, edges: List[List[int]], coins: List[int], k: int) -> int:
        n = len(coins)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        LOG = 15
        def dfs(u, p):
            dp = [-math.inf] * LOG
            dp_childs = [0] * LOG
            child_cnt = 0
            for v in adj[u]:
                if v == p: continue
                child_cnt += 1
                dp_child = dfs(v, u)
                for i in range(LOG):
                    dp_childs[i] += dp_child[i]
            if child_cnt == 0: # initialize everything
                for i in range(LOG):
                    coin = coins[u] // (1 << i)
                    dp[i] = max(coin - k, coin // 2)
            else:
                for i in range(LOG): # not half it
                    coin = coins[u] // (1 << i)
                    dp[i] = max(dp[i], coin - k + dp_childs[i])
                for i in range(LOG): # halved it here
                    coin = coins[u] // (1 << (i + 1))
                    if i + 1 < LOG:
                        dp[i] = max(dp[i], coin + dp_childs[i + 1])
                    else:
                        dp[i] = max(dp[i], coin + dp_childs[i])
            return dp
        res = dfs(0, -1)
        return max(res)
```



# Leetcode Weekly Contest 375

## Count Tested Devices After Test Operations

### Solution 1:  prefix sum

```py
class Solution:
    def countTestedDevices(self, arr: List[int]) -> int:
        n = len(arr)
        psum = res = 0
        for v in arr:
            if v - psum > 0:
                psum += 1
                res += 1
        return res
```

## Double Modular Exponentiation

### Solution 1:  power function with modulus

```py
class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        n = len(variables)
        ans = []
        for i, (a, b, c, m) in enumerate(variables):
            cur = pow(pow(a, b, 10), c, m)
            if cur == target: ans.append(i)
        return ans
```

## Count Subarrays Where Max Element Appears at Least K Times

### Solution 1:  sliding window

```py
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = left = count = 0
        target = max(nums)
        for right in range(n):
            if nums[right] == target: count += 1
            while count > k:
                if nums[left] == target: count -= 1
                left += 1
            if count == k:
                while nums[left] != target:
                    left += 1
                ans += left + 1
        return ans
```

## Count the Number of Good Partitions

### Solution 1:  track the last time each element appears, create blocks, and count the number of ways to choose the blocks, 2^(num_blocks - 1)

```py
class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        mod = int(1e9) + 7
        last = {}
        for i, num in enumerate(nums):
            last[num] = i
        right = num_blocks = 0
        for i, num in enumerate(nums):
            right = max(right, last[num])
            if i == right: num_blocks += 1
        return pow(2, num_blocks - 1, mod)        
```



# Leetcode Weekly Contest 378

## 2980. Check if Bitwise OR Has Trailing Zeros

### Solution 1:  bitwise manipulation, parity

```py
class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        n = len(nums)
        for i in range(n):
            for j in range(i):
                if (nums[i] | nums[j]) % 2 == 0: return True
        return False
```

### Solution 2:  At least two even integers

```py
class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        return len([x for x in nums if x % 2 == 0]) >= 2
```

## 2982. Find Longest Special Substring That Occurs Thrice II

### Solution 1:  max heap, groupby, 26 groups

```py
class Solution:
    def maximumLength(self, s: str) -> int:
        ans = -1
        groups = defaultdict(list)
        for k, g in groupby(s):
            sz = len(list(g))
            heappush(groups[k], -sz)
        for k in groups:
            cnt = 0
            for _ in range(2):
                sz = -heappop(groups[k])
                sz -= 1
                heappush(groups[k], -sz)
            ans = max(ans, -groups[k][0])
        return ans if ans > 0 else -1
```

### Solution 2:  list of every group, 26 groups, sort, take the third largest

```py
class Solution:
    def maximumLength(self, s: str) -> int:
        ans = -1
        n = len(s)
        groups = defaultdict(list)
        groups[s[0]].append(1)
        cur = 1
        for i in range(1, n):
            if s[i] != s[i - 1]:
                cur = 0
            cur += 1
            groups[s[i]].append(cur)
        for lst in groups.values():
            if len(lst) < 3: continue
            lst.sort(reverse = True)
            ans = max(ans, lst[2])
        return ans
```

## 2983. Palindrome Rearrangement Queries

### Solution 1:  ranges, prefix sums, equivalent substrings

```py

```



# Leetcode Weekly Contest 382

## 3019. Number of Changing Keys

### Solution 1:  sum of adjacent diffs

```py
class Solution:
    def countKeyChanges(self, s: str) -> int:
        return sum(1 for i in range(1, len(s)) if s[i - 1].lower() != s[i].lower())
```

## 3020. Find the Maximum Number of Elements in Subset

### Solution 1:  counter, math

```py
class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        MAXN = 10**9
        freq = Counter(nums)
        ans = 0
        for k in sorted(freq):
            cnt = 0
            cur = k
            if cur == 1: 
                cnt += freq[cur]
                freq[cur] = 0
            while cur <= MAXN and freq[cur] > 0:
                cnt += 2
                if freq[cur] == 1: break
                freq[cur] = 0
                cur *= cur
            if cnt % 2 == 0: cnt -= 1
            ans = max(ans, cnt)
        return ans
```

## 3021. Alice and Bob Playing Flower Game

### Solution 1:  math, parity, trick

```py
class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        return sum(m // 2 if i & 1 else (m + 1) // 2 for i in range(1, n + 1))
```

## 3022. Minimize OR of Remaining Elements Using Operations

### Solution 1: 

```py
import operator
class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        MAXB = 30
        n = len(nums)
        mask = ans = 0
        def min_operations(mask):
            cnt = 0
            val = (1 << MAXB) - 1
            for num in nums:
                if (num & mask) != 0:
                    val &= (num & mask)
                    if val == 0: 
                        val = (1 << MAXB) - 1
                        continue
                    cnt += 1
                else:
                    val = (1 << MAXB) - 1
            return cnt
        for i in reversed(range(MAXB)):
            x = min_operations(mask | (1 << i))
            if x <= k: mask |= (1 << i)
        return reduce(operator.or_, [(1 << i) for i in range(MAXB) if not ((mask >> i) & 1)], 0)
```



# Leetcode Weekly Contest 383

## 3028. Ant on the Boundary

### Solution 1:  simulation

```py
class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        ans = pos = 0
        for num in nums:
            pos += num
            if pos == 0: ans += 1
        return ans
```

## 3030. Find the Grid of Region Average

### Solution 1: 

```py

```

## 3031. Minimum Time to Revert Word to Initial State II

### Solution 1:  KMP and prefix array algorithm, divisibility

the index i of the suffix that matches prefix is at i = n - j.  That needs to be divisible by k, 
cause give abacaba and k = 3
you are doing this, cabaxxx at first operation
axxxxxx at the second operation, just need a to match prefix, cause xxxxxxx can be assigned any character we desired.  So easy to make it match the original string. 

```py
def kmp(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]: 
            j = pi[j - 1]
        if s[j] == s[i]: j += 1
        pi[i] = j
    return pi

class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        prefix_arr = kmp(word)
        j = prefix_arr[-1]
        while j > 0 and (n - j) % k != 0:
            j = prefix_arr[j - 1]
        return math.ceil((n - j) / k)
```



# Leetcode Weekly Contest 384

## 3033. Modify the Matrix

### Solution 1:  max column

```py
class Solution:
    def modifiedMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        R, C = len(mat), len(mat[0])
        maxcol = [0] * C
        for r, c in product(range(R), range(C)):
            maxcol[c] = max(maxcol[c], mat[r][c])
        for r, c in product(range(R), range(C)):
            if mat[r][c] == -1: mat[r][c] = maxcol[c]
        return mat
```

## 3035. Maximum Palindromes After Operations

### Solution 1:  sort, greedy, count pairs and singles of characters

```py
class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        n = len(words)
        ans = 0
        freq = Counter()
        for w in words: freq.update(Counter(w))
        pairs = sum(v // 2 for v in freq.values())
        singles = sum(1 for v in freq.values() if v & 1)
        for w in sorted(words, key = len):
            sz = len(w)
            if sz & 1:
                if singles > 0: singles -= 1
                else: pairs -= 1; singles += 1
            if pairs >= sz // 2:
                pairs -= sz // 2
                ans += 1
        return ans
```

## 3036. Number of Subarrays That Match a Pattern II

### Solution 1:  z algorithm, pattern matching

```py
def z_algorithm(s) -> list[int]:
    n = len(s)
    z = [0]*n
    left = right = 0
    for i in range(1,n):
        # BEYOND CURRENT MATCHED SEGMENT, TRY TO MATCH WITH PREFIX
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
            # IF PREVIOUS MATCHED SEGMENT TOUCHES OR PASSES THE RIGHT BOUNDARY OF CURRENT MATCHED SEGMENT
            else:
                left = i
                while right < n and s[right-left] == s[right]:
                    right += 1
                z[i] = right - left
                right -= 1
    return z
class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n, m = len(nums), len(pattern)
        diff = [0] * (n - 1)
        for i in range(n - 1):
            if nums[i + 1] > nums[i]: diff[i] = 1
            elif nums[i + 1] < nums[i]: diff[i] = -1
        encoded = pattern + [2] + diff
        z_arr = z_algorithm(encoded)
        ans = sum(1 for x in z_arr if x == m)
        return ans
```

### Solution 2:  Rolling hash

```py
class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n, m = len(nums), len(pattern)
        p, MOD = 31, int(1e9)+7
        coefficient = lambda x: x + 2
        pat_hash = 0
        for v in pattern:
            pat_hash = (pat_hash * p + coefficient(v)) % MOD
        diff = [0] * (n - 1)
        for i in range(n - 1):
            if nums[i + 1] > nums[i]: diff[i] = 1
            elif nums[i + 1] < nums[i]: diff[i] = -1
        POW = 1
        for _ in range(m - 1):
            POW = (POW * p) % MOD
        ans = cur_hash = 0
        for i, v in enumerate(diff):
            cur_hash = (cur_hash * p + coefficient(v)) % MOD
            if i >= m - 1:
                if cur_hash == pat_hash: ans += 1
                cur_hash = (cur_hash - coefficient(diff[i - m + 1]) * POW) % MOD
        return ans
```

### Solution 3:  KMP algorithm, prefix array

```py
def kmp(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]: 
            j = pi[j - 1]
        if s[j] == s[i]: j += 1
        pi[i] = j
    return pi
class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n, m = len(nums), len(pattern)
        diff = [0] * (n - 1)
        for i in range(n - 1):
            if nums[i + 1] > nums[i]: diff[i] = 1
            elif nums[i + 1] < nums[i]: diff[i] = -1
        encoded = pattern + [2] + diff
        parr = kmp(encoded)
        ans = sum(1 for x in parr if x == m)
        return ans
```



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

# Leetcode Weekly Contest 386

## 3047. Find the Largest Area of Square Inside Two Rectangles

### Solution 1: 

```py
class Solution:
    def largestSquareArea(self, bottomLeft: List[List[int]], topRight: List[List[int]]) -> int:
        n = len(bottomLeft)
        ans = 0
        intersection = lambda s1, e1, s2, e2: min(e1, e2) - max(s1, s2)
        for i in range(n):
            (x1, y1), (x2, y2) = bottomLeft[i], topRight[i]
            for j in range(i + 1, n):
                (x3, y3), (x4, y4) = bottomLeft[j], topRight[j]
                s = max(0, min(intersection(x1, x2, x3, x4), intersection(y1, y2, y3, y4)))
                ans = max(ans, s * s)
        return ans
```

## 3048. Earliest Second to Mark Indices I

### Solution 1:  sortedlist, greedy, simulation, O(nmlog(n))

```py
from sortedcontainers import SortedList
class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        changeIndices = [x - 1 for x in changeIndices]
        n, m = len(nums), len(changeIndices)
        index = [[] for _ in range(n)]
        marked = [0] * n
        for i in range(m):
            index[changeIndices[i]].append(i)
        sl = SortedList()
        for i in range(n):
            try:
                marked[i] = index[i].pop()
                sl.add((marked[i], i))
            except:
                return -1
        def search():
            for i in reversed(range(m)):
                prev = bag = 0
                for x, j in sl:
                    delta = x - prev
                    bag += delta
                    bag -= nums[j]
                    prev = x + 1
                    if bag < 0: return i + 2
                idx = changeIndices[i]
                sl.remove((marked[idx], idx))
                # need to process in order
                try:
                    marked[idx] = index[idx].pop()
                    sl.add((marked[idx], idx))
                except:
                    return i + 1
            return -1
        ans = search()
        return ans if 0 <= ans <= m else -1
```

### Solution 2:  binary search, O((n + m)log(m)), 

FFFFFT, return T

```py
class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        changeIndices = [x - 1 for x in changeIndices]
        n, m = len(nums), len(changeIndices)
        left, right = 0, m + 1
        def possible(target):
            marked = [-1] * n
            for i in range(target):
                marked[changeIndices[i]] = i
            if any(m == -1 for m in marked): return False
            prev = bag = 0
            for i in sorted(range(n), key = lambda i: marked[i]):
                delta = marked[i] - prev
                bag += delta
                bag -= nums[i]
                prev = marked[i] + 1
                if bag < 0: return False
            return True
        while left < right:
            mid = (left + right) >> 1
            if possible(mid):
                right = mid
            else:
                left = mid + 1
        return left if left <= m else -1
```

## 3048. Earliest Second to Mark Indices II

### Solution 1: 

```py

```



# Leetcode Weekly Contest 387

## 3070. Count Submatrices with Top-Left Element and Sum Less Than k

### Solution 1:  columnwise 2d prefix sum

```py
class Solution:
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        R, C = len(grid), len(grid[0])
        ps = [[0] * C for _ in range(R)]
        # build columnwise prefix sum
        for r, c in product(range(R), range(C)):
            ps[r][c] = grid[r][c]
            if r > 0: ps[r][c] += ps[r - 1][c]
        ans = 0
        for r in range(R):
            sum_ = 0
            for c in range(C):
                sum_ += ps[r][c]
                if sum_ > k: break
                ans += 1
        return ans
```

## 3071. Minimum Operations to Write the Letter Y on a Grid

### Solution 1:  counter, matrix

```py
class Solution:
    def minimumOperationsToWriteY(self, grid: List[List[int]]) -> int:
        N = len(grid)
        ycounts, counts = [0] * 3, [0] * 3
        for r, c in product(range(N), repeat = 2):
            if (r >= N // 2 and c == N // 2) or (r < N // 2 and c in (r, N - r - 1)): ycounts[grid[r][c]] += 1
            else: counts[grid[r][c]] += 1
        ysum, sum_ = sum(ycounts), sum(counts)
        ans = math.inf
        for i, j in product(range(3), repeat = 2):
            if i == j: continue # set y to be i, and complement of y to be j
            ans = min(ans, ysum - ycounts[i] + sum_ - counts[j])
        return ans
```

## 3072. Distribute Elements Into Two Arrays II

### Solution 1:  coordinate compression fenwick tree, range counts queries

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
    def resultArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        coords = {}
        for v in sorted(nums):
            if v not in coords: coords[v] = len(coords) + 1
        m = len(coords)
        fenwicks = list(map(lambda _: FenwickTree(m), range(2)))
        arrs = [[] for _ in range(2)]
        for i in range(2):
            arrs[i].append(nums[i])
            fenwicks[i].update(coords[nums[i]], 1)
        for val in nums[2:]:
            l, r = coords[val], len(coords)
            counts = [fenwicks[i].query_range(l + 1, r) for i in range(2)]
            if counts[0] != counts[1]:
                idx = int(counts[0] < counts[1])
            else:
                idx = int(len(arrs[0]) > len(arrs[1]))
            arrs[idx].append(val)
            fenwicks[idx].update(l, 1)
        return sum(arrs, [])
```



# Leetcode Weekly Contest 388

## 3075. Maximize Happiness of Selected Children

### Solution 1:  sort, max, sum, greedy

```py
class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse = True)
        return sum(max(0, happiness[i] - i) for i in range(k))
```

## 3076. Shortest Uncommon Substring in an Array

### Solution 1:  brute force, counter

```py
class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        counts = Counter()
        n = len(arr)
        ans = [""] * n
        for word in arr:
            nw = len(word)
            for i in range(nw + 1):
                for j in range(i):
                    counts[word[j : i]] += 1
        for i in range(n):
            nw = len(arr[i])
            for j in range(nw + 1):
                for k in range(j):
                    counts[arr[i][k : j]] -= 1
            for len_ in range(1, nw + 1):
                for l in range(nw - len_ + 1):
                    r = l + len_
                    cur = arr[i][l : r]
                    if counts[cur] == 0 and (not ans[i] or cur < ans[i]):
                        ans[i] = cur
                if ans[i]: break
            for j in range(nw + 1):
                for k in range(j):
                    counts[arr[i][k : j]] += 1
        return ans
```

## 3077. Maximum Strength of K Disjoint Subarrays

### Solution 1: dp, O(2*n*k)

```py
class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[-math.inf] * 2 for _ in range(k + 1)] # (subarray, started)
        # 0 not started, 1 started
        dp[0][0] = 0
        for num in nums:
            ndp = [[-math.inf] * 2 for _ in range(k + 1)]
            for i in range(k + 1):
                # continuation of a subarray
                score = num * (k - i) * (1 if i % 2 == 0 else -1)
                ndp[i][1] = max(ndp[i][1], dp[i][1] + score)
                # start new subarray
                if i > 0: ndp[i][1] = max(ndp[i][1], dp[i - 1][1] + score)
                ndp[i][1] = max(ndp[i][1], dp[i][0] + score)
                # skip element and section
                if i > 0: ndp[i][0] = max(ndp[i][0], dp[i - 1][1])
                ndp[i][0] = max(ndp[i][0], dp[i][0])
            dp = ndp
        return max(dp[-2][1], dp[-1][0])
```



# Leetcode Weekly Contest 389

## 3084. Count Substrings Starting and Ending with Given Character

### Solution 1: counting, number of ways to pick 2 elements from n

```py
class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        n = s.count(c)
        return n * (n + 1) // 2
```

## 3085. Minimum Deletions to Make String K-Special

### Solution 1: frequency array, precompute prefix, greedy

```py
class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        n = len(word)
        unicode = lambda ch: ord(ch) - ord("a")
        freq = [0] * 26
        for ch in word:
            freq[unicode(ch)] += 1
        ans = n
        cur = 0
        freq.sort()
        for f in freq:
            if f == 0: continue 
            take = 0
            for j in range(26):
                if freq[j] < f: continue
                delta = max(0, freq[j] - f - k)
                take += delta
            ans = min(ans, cur + take)
            cur += f
        return ans
```

## 3086. Minimum Moves to Pick K Ones

### Solution 1:  prefix sum, rolling median deviation, greedy

```py
class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ans = math.inf
        nums = [i for i, x in enumerate(nums) if x]
        n = len(nums)
        psum = list(accumulate(nums))
        def sum_(i, j):
            return psum[j] - (psum[i - 1] if i > 0 else 0)
        def deviation(i, j, mid):
            lsum = (mid - i + 1) * nums[mid] - sum_(i, mid)
            rsum = sum_(mid, j) - (j - mid + 1) * nums[mid]
            return lsum + rsum
        def RMD(nums, k): # rolling median deviation
            if k == 0: return 0
            ans = math.inf
            l = 0
            for r in range(k - 1, n):
                mid = (l + r) >> 1
                ans = min(ans, deviation(l, r, mid))
                if k % 2 == 0:
                    ans = min(ans, deviation(l, r, mid + 1))
                l += 1
            return ans
        L, R = max(0, min(k, maxChanges) - 3), min(k, maxChanges)
        for m in range(L, R + 1):
            ans = min(ans, RMD(nums, k - m) + 2 * m)
        return ans
```



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



# Leetcode Weekly Contest 391

## 3101. Count Alternating Subarrays

### Solution 1: sliding window

```py
class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        delta = ans = 0
        prev = None
        for x in nums:
            if x == prev:
                delta = 0
            delta += 1
            ans += delta
            prev = x
        return ans
```

## 3102. Minimize Manhattan Distances

### Solution 1:  maximum manhattan disatnce for pair of points

```py
class Solution:
    def max_manhattan_distance(self, points, remove = -1):
        smin = dmin = math.inf
        smax = dmax = -math.inf
        smax_i = smin_i = dmax_i = dmin_i = None
        for i, (x, y) in enumerate(points):
            if remove == i: continue
            s = x + y
            d = x - y
            if s > smax:
                smax = s
                smax_i = i
            if s < smin:
                smin = s
                smin_i = i
            if d > dmax:
                dmax = d
                dmax_i = i
            if d < dmin:
                dmin = d
                dmin_i = i
        return (smax_i, smin_i) if smax - smin >= dmax - dmin else (dmax_i, dmin_i)
    def minimumDistance(self, points: List[List[int]]) -> int:
        i, j = self.max_manhattan_distance(points)
        manhattan_distance = lambda i, j: abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
        return min(
            manhattan_distance(*self.max_manhattan_distance(points, i)),
            manhattan_distance(*self.max_manhattan_distance(points, j))
        )
```

# Leetcode Weekly Contest 392

## 3107. Minimum Operations to Make Median of Array Equal to K

### Solution 1:  binary search, median

```py
class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2
        nums.sort()
        i = bisect_left(nums, k)
        ans = 0
        if i > m:
            for j in range(m, i):
                ans += k - nums[j]
        else:
            for j in range(i, m + 1):
                ans += nums[j] - k
        return ans
```

## 3108. Minimum Cost Walk in Weighted Graph

### Solution 1:  union find, bitwise and operation

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
class Solution:
    def minimumCost(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
        BITS = 18
        dsu = UnionFind(n)
        for u, v, _ in edges:
            dsu.union(u, v)
        resp = defaultdict(lambda: (1 << BITS) - 1)
        for u, v, w in edges:
            root = dsu.find(u)
            resp[root] &= w
        m = len(query)
        ans = [-1] * m
        for i, (s, t) in enumerate(query):
            if dsu.find(s) != dsu.find(t): continue
            ans[i] = resp[dsu.find(t)] if s != t else 0
        return ans
```

# Leetcode Weekly Contest 393

## 3116. Kth Smallest Amount With Single Denomination Combination

### Solution 1:  binary search, inclusion exclusion principle, lcm

```py
class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        n = len(coins)
        lcms = [[] for _ in range(n + 1)]
        for mask in range(1, 1 << n):
            cur = 1
            for i in range(n):
                if (mask >> i) & 1:
                    cur = math.lcm(cur, coins[i])
            lcms[mask.bit_count()].append(cur)
        def possible(target):
            ans = 0
            for i in range(1, n + 1):
                for v in lcms[i]:
                    if i & 1: ans += target // v
                    else: ans -= target // v
            return ans
        l, r = 0, 25 * 2 * 10 ** 9
        while l < r:
            m = (l + r) >> 1
            if possible(m) < k:
                l = m + 1
            else:
                r = m
        return l
```

## 3117. Minimum Sum of Values by Dividing Array

### Solution 1:  range bitwise or queries with sparse table, binary search, dynamic programming, min heap, line sweep

```py
class ST_And:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 14 # 10,000
        self.build()

    def op(self, x, y):
        return x & y

    def build(self):
        self.lg = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.lg[i] = self.lg[i // 2] + 1
        self.st = [[0] * self.n for _ in range(self.LOG)]
        for i in range(self.n):
            self.st[0][i] = self.nums[i]
        # CONSTRUCT SPARSE TABLE
        for i in range(1, self.LOG):
            j = 0
            while (j + (1 << (i - 1))) < self.n:
                self.st[i][j] = self.op(self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))])
                j += 1

    def query(self, l, r):
        length = r - l + 1
        i = self.lg[length]
        return self.op(self.st[i][l], self.st[i][r - (1 << i) + 1])
class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        st_and = ST_And(nums)
        def ubsearch(start, target):
            l, r = start, n - 1
            while l < r:
                m = (l + r + 1) >> 1
                if st_and.query(start, m) >= target:
                    l = m
                else:
                    r = m - 1
            return l if st_and.query(start, l) == target else -1
        def lbsearch(start, target):
            l, r = start, n - 1
            while l < r:
                m = (l + r) >> 1
                if st_and.query(start, m) > target:
                    l = m + 1
                else:
                    r = m
            return l if st_and.query(start, l) == target else n
        dp = [[math.inf] * m for _ in range(n)]
        target = andValues[0]
        s, e = lbsearch(0, target), ubsearch(0, target)
        if s > e: return -1
        for i in range(s, e + 1):
            dp[i][0] = nums[i]
        for j in range(1, m):
            events = [(math.inf, math.inf)] * n
            activate = [(math.inf, math.inf)] * n
            minheap = []
            target = andValues[j]
            for i in range(1, n):
                if dp[i - 1][j - 1] == math.inf: continue
                s, e = lbsearch(i, target), ubsearch(i, target)
                if s > e: continue
                events[i] = (s, e)
            found = False
            for i in range(1, n):
                s, e = events[i]
                if s < math.inf: activate[s] = min(activate[s], (dp[i - 1][j - 1], e))
                if activate[i] is not None: heappush(minheap, activate[i])
                while minheap and minheap[0][1] < i: heappop(minheap)
                if minheap: 
                    dp[i][j] = minheap[0][0] + nums[i]
                    found = True
            if not found: return -1
        return dp[-1][-1] if dp[-1][-1] < math.inf else -1
```

### Solution 2:  range bitwise or queries with sparse table, binary search, range minimum query with sparse table, dynamic programming

```py
class ST_And:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 14 # 10,000
        self.build()

    def op(self, x, y):
        return x & y

    def build(self):
        self.lg = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.lg[i] = self.lg[i // 2] + 1
        self.st = [[0] * self.n for _ in range(self.LOG)]
        for i in range(self.n):
            self.st[0][i] = self.nums[i]
        # CONSTRUCT SPARSE TABLE
        for i in range(1, self.LOG):
            j = 0
            while (j + (1 << (i - 1))) < self.n:
                self.st[i][j] = self.op(self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))])
                j += 1

    def query(self, l, r):
        length = r - l + 1
        i = self.lg[length]
        return self.op(self.st[i][l], self.st[i][r - (1 << i) + 1])
class ST_Min:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 14 # 10,000
        self.build()

    def op(self, x, y):
        return min(x, y)

    def build(self):
        self.lg = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.lg[i] = self.lg[i // 2] + 1
        self.st = [[0] * self.n for _ in range(self.LOG)]
        for i in range(self.n):
            self.st[0][i] = self.nums[i]
        # CONSTRUCT SPARSE TABLE
        for i in range(1, self.LOG):
            j = 0
            while (j + (1 << (i - 1))) < self.n:
                self.st[i][j] = self.op(self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))])
                j += 1

    def query(self, l, r):
        length = r - l + 1
        i = self.lg[length]
        return self.op(self.st[i][l], self.st[i][r - (1 << i) + 1])
class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        st_and = ST_And(nums)
        def ubsearch(end, target):
            l, r = 0, end
            while l < r:
                m = (l + r + 1) >> 1
                if st_and.query(m, end) <= target:
                    l = m
                else:
                    r = m - 1
            return l if st_and.query(l, end) == target else -1
        def lbsearch(end, target):
            l, r = 0, end
            while l < r:
                m = (l + r) >> 1
                if st_and.query(m, end) < target:
                    l = m + 1
                else:
                    r = m
            return l if st_and.query(l, end) == target else n
        dp = [[math.inf] * n for _ in range(m)]
        # INITIALIZE
        target = andValues[0]
        for i in range(n):
            l, r = lbsearch(i, target), ubsearch(i, target)
            if l > r: continue
            dp[0][i] = nums[i]
        # MAIN DP ITERATIONS
        for j in range(1, m):
            st_min = ST_Min(dp[j - 1])
            target = andValues[j]
            for i in range(1, n):
                l, r = max(0, lbsearch(i, target) - 1), max(0, ubsearch(i, target) - 1)
                if l > r: continue
                dp[j][i] = st_min.query(l, r) + nums[i]
        return dp[-1][-1] if dp[-1][-1] < math.inf else -1
```



# Leetcode Weekly Contest 394

## 100290. Minimum Number of Operations to Satisfy Conditions

### Solution 1:  dynamic programming with pmin and smin, frequency count for each column

```py
class Solution:
    def minimumOperations(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        dp = [0] * 10
        for c in range(C):
            freq = [0] * 10
            for r in range(R):
                freq[grid[r][c]] += 1
            pmin = [math.inf] * 10
            smin = [math.inf] * 10
            for i in range(10):
                pmin[i] = dp[i]
                if i > 0: pmin[i] = min(pmin[i], pmin[i - 1])
            for i in reversed(range(10)):
                smin[i] = dp[i]
                if i < 9: smin[i] = min(smin[i], smin[i + 1])
            for i, f in enumerate(freq):
                dp[i] = R - f
                mn = math.inf
                if i > 0:
                    mn = min(mn, pmin[i - 1])
                if i < 9:
                    mn = min(mn, smin[i + 1])
                if mn != math.inf: dp[i] += mn
        return min(dp)
```

## 100276. Find Edges in Shortest Paths

### Solution 1:  double dijkstra algorithm

```py
def dijkstra(adj, src):
    N = len(adj)
    min_heap = [(0, src)]
    dist = [math.inf] * N
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if cost >= dist[u]: continue
        dist[u] = cost
        for v, w in adj[u]:
            if cost + w < dist[v]: heapq.heappush(min_heap, (cost + w, v))
    return dist

class Solution:
    def findAnswer(self, n: int, edges: List[List[int]]) -> List[bool]:
        m = len(edges)
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        src_dist = dijkstra(adj, 0)
        dst_dist = dijkstra(adj, n - 1)
        target = src_dist[n - 1]
        ans = [False] * m
        if target == math.inf: return ans
        for i, (u, v, w) in enumerate(edges):
            ans[i] = src_dist[u] + w + dst_dist[v] == target or src_dist[v] + w + dst_dist[u] == target
        return ans
```


# Leetcode Weekly Contest 396

## 3138. Minimum Length of Anagram Concatenation

### Solution 1:  factorization, highly composite number, frequency count

```py
class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        div = []
        for x in range(1, int(math.sqrt(n)) + 1):
            if n % x == 0: 
                div.append(x)
                if n // x != x: div.append(n // x)
        div.sort()
        unicode = lambda ch: ord(ch) - ord("a")
        def possible(target):
            freq = [0] * 26
            for i in range(target):
                freq[unicode(s[i])] += 1
            for i in range(target, n):
                if i % target == 0: nfreq = [0] * 26
                v = unicode(s[i])
                nfreq[v] += 1
                if nfreq[v] > freq[v]: return False
            return True
        for x in div:
            if possible(x): return x
        return n
```

## 3139. Minimum Cost to Equalize Array

### Solution 1: 

```py

```

# Leetcode Weekly Contest 397

## 3147. Taking Maximum Energy From the Mystic Dungeon

### Solution 1:  dynamic programming, kadane's algorithm, take largest amongst last k

```py
class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n
        for i in range(n):
            dp[i] = max(energy[i], energy[i] + (dp[i - k] if i >= k else 0))
        return max(dp[-k:])
```

## 3148. Maximum Difference Score in a Grid

### Solution 1:  dynamic programming, matrix, row and column max

```py
class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        ans = -math.inf
        rmax = [-math.inf] * R
        cmax = [-math.inf] * C
        for r, c in product(range(R), range(C)):
            mx = max(rmax[r], cmax[c])
            cur = mx + grid[r][c]
            ans = max(ans, cur)
            cur = max(-grid[r][c], mx)
            rmax[r] = max(rmax[r], cur)
            cmax[c] = max(cmax[c], cur)
        return ans
```

## 3149. Find the Minimum Cost Array Permutation

### Solution 1:  recursive, bitmask, dp, tracking

```py
class Solution:
    def findPermutation(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ptr = [[-1] * n for _ in range(1 << n)]
        dp = [[math.inf] * n for _ in range(1 << n)]
        def dfs(mask, p):
            if mask.bit_count() == n: return abs(nums[0] - p)
            if dp[mask][p] == math.inf:
                for i in range(n):
                    if (mask >> i) & 1: continue
                    val = abs(nums[i] - p) + dfs(mask | (1 << i), i)
                    if val < dp[mask][p]:
                        dp[mask][p] = val
                        ptr[mask][p] = i
            return dp[mask][p]
        dfs(1, 0)
        ans = [0]
        mask = 1
        for _ in range(n - 1):
            ans.append(ptr[mask][ans[-1]])
            mask |= (1 << ans[-1])
        return ans
```

# Leetcode Weekly Contest 398

## 3152. Special Array II

### Solution 1:  sort, offline query, line sweep

```py
class Solution:
    def isArraySpecial(self, nums: List[int], queries: List[List[int]]) -> List[bool]:
        n = len(nums)
        q = len(queries)
        ans = [True] * q
        events, on_stack = [], []
        for i, (l, r) in enumerate(queries):
            events.append((l, i, 0))
            events.append((r, i, 1))
        events.sort()
        idx = 0
        vis = [0] * q
        for i in range(n):
            if i > 0 and nums[i] % 2 == nums[i - 1] % 2:
                while on_stack:
                    pr = on_stack.pop()
                    if vis[pr]: ans[pr] = False
            while idx < len(events) and events[idx][0] == i:
                s, j, ev = events[idx]
                if ev == 0: 
                    on_stack.append(j)
                    vis[j] = 1
                else: vis[j] = 0 # no longer visited
                idx += 1
        return ans
```

## 3153. Sum of Digit Differences of All Pairs

### Solution 1:  frequency of digit at each position

```py
class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        m = len(str(nums[0]))
        freq = [[0] * 10 for _ in range(m)]
        for num in nums:
            for i, dig in enumerate(map(int, str(num))):
                freq[i][dig] += 1
        ans = 0
        for num in nums:
            for i, dig in enumerate(map(int, str(num))):
                for f in range(10):
                    if f == dig: continue
                    ans += freq[i][f]
                freq[i][dig] -= 1
        return ans
```

## 3154. Find Number of Ways to Reach the K-th Stair

### Solution 1:  dynamic programming, count

```py
class Solution:
    def waysToReachStair(self, k: int) -> int:
        dp = Counter({(1, 0, 0): 1})
        ans = 0
        while dp:
            ndp = Counter()
            for (a, b, c), v in dp.items():
                if a > k + 5: continue
                if a == k: ans += v
                if c == 0: ndp[(a - 1, b, 1)] += v
                ndp[(a + 2 ** b, b + 1, 0)] += v
            dp = ndp
        return ans
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
from sortedcontainers import Sortedlist
class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        ans = cur = start = 0
        is_pos = False
        blocks = SortedList()
        podd, peven = FenwickTree(n), FenwickTree(n)
        for i in range(n):
            if i & 1: podd.update(i + 1, nums[i])
            else: peven.update(i + 1, nums[i])
            if nums[i] >= 0: cur += nums[i]
            if i == 0:
                is_pos = nums[i] >= 0
            if nums[i] < 0 and is_pos:
                blocks.add((start, i))
                is_pos = False
                start = i
            elif nums[i] >= 0 and not is_pos:
                blocks.add((start, i))
                is_pos = True
                start = i
        blocks.add((start, n))
        def update(i, x):
            if i & 1:
                podd.update(i + 1, -nums[i])
                podd.update(i + 1, x)
            else:
                peven.update(i + 1, -nums[i])
                peven.update(i + 1, x)
        print(blocks)
        for pos, x in queries:
            idx = blocks.bisect_left((pos, pos))
            l, r = blocks[idx]
            si = podd.query_range(l + 1, r) >= 0 or peven.query_ange(l + 1, r) >= 0
            if si:
                if x >= 0:
                    cur -= max(podd.query_range(l + 1, r), peven.query_range(l + 1, r))
                    update(pos, x)
                    cur += max(podd.query_range(l + 1, r), peven.query_range(l + 1, r))
                else:
                    cur -= max(podd.query_range(l + 1, r), peven.query_range(l + 1, r))
                    m = pos 
                    blocks.pop(idx)
                    blocks.add((l, m))
                    blocks.add((m + 1, r))
                    blocks.add((m, m + 1))
                    update(pos, x)
                    cur += max(podd.query_range(l + 1, m), peven.query_range(l + 1, m)) + max(podd.query_range(m + 1, r), peven.query_range(m + 1, r))
            else:
                if x < 0: update(pos, x)
                else:
                   m = pos
                   blocks.pop(idx)
                   blocks.add((l, m))
                   blocks.add((m + 1, r))
                   blocks.add((m, m + 1))
                   update(pos, x)
                   cur +=  

            nums[i] = x
            ans += cur
        return ans
```