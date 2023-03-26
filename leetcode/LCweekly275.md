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