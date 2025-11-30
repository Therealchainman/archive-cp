# Prefix Tree

# Trie datastructure that is very useful for prefix matching

This inherits the defaultdict class which makes it easier to work with trie data structure.  This specifict one is able to count the number of words that have a certain prefix.
The benefit of this is that you can add, remove and query the trie in O(n) time where n is the length of the longest word in the trie.

remove can be done by decrementing the prefix_count of each prefix for a word that is removed.


```py
class TrieNode(defaultdict):
    def __init__(self):
        super().__init__(TrieNode)
        self.prefix_count = 0 # how many words have this prefix

    def __repr__(self) -> str:
        return f'is_word: {self.is_word} prefix_count: {self.prefix_count}, children: {self.keys()}'
```

## reduce trie

This is a trie with the use of reduce to add words to the trie, and you can search through the trie like normal. 

```py
TrieNode = lambda: defaultdict(TrieNode)
root = TrieNode()
# adding words into the trie data structure
for word in dictionary:
    reduce(dict.__getitem__, word, root)['word'] = True


    
for word in sentence.split():
    cur = root
    n = len(word)
    for i in range(n):
        cur = cur[word[i]]
        if cur['word']:
```

## Example of using Trie to find prefix and word count in a trie data structure

This is useful if you want to find the number of words that have a certain prefix.  This is useful for autocomplete and other things.

```py
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.prefix_count = self.word_count = 0

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children[ch]
            node.prefix_count += 1
        node.word_count += 1

    def countWordsEqualTo(self, word: str) -> int:
        node = self.root
        for ch in word:
            node = node.children[ch]
        return node.word_count

    def countWordsStartingWith(self, prefix: str) -> int:
        node = self.root
        for ch in prefix:
            node = node.children[ch]
        return node.prefix_count

    def erase(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children[ch]
            node.prefix_count -= 1
        node.word_count -= 1
```

## Prefix Tree (Trie) for strings in C++

This is an implementation of a Trie (prefix tree) specialized for lowercase English letters (`'a'` to `'z'`).  
It efficiently supports insertion, prefix queries, counting prefixes, and erasing prefixes with adjustable counts.

---

## Structures

### `struct Node`

Represents a single node in the Trie.

- **Members:**
  - `int children[26]`  
    Array holding indices of child nodes for each character `'a'` to `'z'`.
    If `children[i] == 0`, there is no child for character `('a' + i)`.
  - `bool isLeaf`  
    Indicates if the node marks the end of a word.
  - `int cnt`  
    Number of words that pass through or end at this node.

- **Methods:**
  - `void init()`  
    Initializes the node:
    - Sets all `children` to 0 (no children).
    - Sets `isLeaf` to `false`.
    - Sets `cnt` to `0`.

---

### `struct Trie`

Represents the entire Trie structure.

- **Members:**
  - `vector<Node> trie`  
    Dynamic array of nodes forming the Trie.

- **Methods:**
  - `void init()`  
    Initializes the Trie with a root node.
  
  - `void insert(const string& s)`  
    Inserts a string `s` into the Trie:
    - Creates new nodes as needed.
    - Increments `cnt` along the path.
    - Marks the end node with `isLeaf = true`.

  - `bool startsWith(const string& s)`  
    Checks if a prefix `s` exists **and** is a complete word in the Trie:
    - Traverses the nodes for each character.
    - Returns `true` if it encounters a `isLeaf` node after following `s`.

  - `int countPrefix(const string& s)`  
    Counts how many words in the Trie have the prefix `s`:
    - Traverses the prefix path.
    - Returns the `cnt` at the last node.
  
  - `void eraseCount(const string& s, int val)`  
    Decreases the count `cnt` by `val` along the prefix path of `s`, and recursively clears all descendants' `cnt` values if needed:
    - Useful for batch deletion or invalidation of words starting with a certain prefix.

---

## Notes

- The Trie supports only **lowercase English letters** (`'a'` to `'z'`).
- The `children` array uses **0-based indexing**, with `'a'` corresponding to index `0`, `'b'` to `1`, and so on.
- `cnt` tracks how many words pass through a node, allowing efficient prefix counting and deletions.
- New nodes are dynamically allocated and indexed by the `vector`'s size.

```cpp
struct Node {
    int children[26];
    bool isLeaf;
    int cnt;
    void init() {
        memset(children, 0, sizeof(children));
        isLeaf = false;
        cnt = 0;
    }
};
struct Trie {
    vector<Node> trie;
    void init() {
        Node root;
        root.init();
        trie.emplace_back(root);
    }
    void insert(const string& s) {
        int cur = 0;
        for (const char &c : s) {
            int i = c - 'a';
            if (trie[cur].children[i]==0) {
                Node root;
                root.init();
                trie[cur].children[i] = trie.size();
                trie.emplace_back(root);
            }
            cur = trie[cur].children[i];
            trie[cur].cnt++;
        }
        trie[cur].isLeaf= true;
    }
    bool startsWith(const string& s) {
        int cur = 0;
        for (const char &c : s) {
            int i = c - 'a';
            if (!trie[cur].children[i]) return false;
            cur = trie[cur].children[i];
            if (trie[cur].isLeaf) return true;
        }
        return false;
    }
    int countPrefix(const string& s) {
        int cur = 0;
        for (const char &c : s) {
            int i = c - 'a';
            if (!trie[cur].children[i]) return 0;
            cur = trie[cur].children[i];
        }
        return trie[cur].cnt;
    }
    void eraseCount(const string& s, int val) {
        int cur = 0;
        for (const char &c : s) {
            int i = c - 'a';
            if (!trie[cur].children[i]) return;
            cur = trie[cur].children[i];
            trie[cur].cnt -= val;
        }
        queue<int> q;
        q.emplace(cur);
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            trie[node].cnt = 0;
            for (int i = 0; i < 26; i++) {
                if (trie[node].children[i] && trie[trie[node].children[i]].cnt) {
                    q.emplace(trie[node].children[i]);
                }
            }
        }
    }

    // add erase method sometime? trie can support erase
};

Trie trie;
trie.init();
```


## A prefix tree with dfs and merging tries into one

```cpp
struct Node {
    int children[26];
    bool isLeaf;
    int64 cnt;
    void init() {
        memset(children, 0, sizeof(children));
        isLeaf = false;
        cnt = 0;
    }
};
struct Trie {
    vector<Node> trie;
    void init() {
        Node root;
        root.init();
        trie.emplace_back(root);
    }
    void insert(const vector<vector<pair<int, char>>>& adj, int u = 0, int cur = 0) {
        trie[cur].cnt++;
        for (const auto& [v, c] : adj[u]) {
            int idx = c - 'a';
            if (trie[cur].children[idx] == 0) {
                Node node;
                node.init();
                trie[cur].children[idx] = trie.size();
                trie.emplace_back(node);
            }
            insert(adj, v, trie[cur].children[idx]);
        }
    }
    int64 dfs(int cur) {
        int64 ans = choose3(N) - choose3(N - trie[cur].cnt);
        for (int i = 0; i < 26; ++i) {
            if (trie[cur].children[i]) {
                ans += dfs(trie[cur].children[i]);
            }
        }
        return ans;
    }
};
```

example where we need the index, and use multiset to store common prefix of some length above K. 

This uses multiset, and erase

```cpp
int K;
multiset<int, greater<int>> lengths; // max multiset, descending order
struct Node {
    int children[26];
    int cnt;
    void init() {
        memset(children, 0, sizeof(children));
        cnt = 0;
    }
};
struct Trie {
    vector<Node> trie;
    void init() {
        Node root;
        root.init();
        trie.emplace_back(root);
    }
    void insert(const string& s) {
        int cur = 0;
        for (int i = 0; i < s.size(); ++i) {
            int cv = s[i] - 'a';
            if (trie[cur].children[cv]==0) {
                Node root;
                root.init();
                trie[cur].children[cv] = trie.size();
                trie.emplace_back(root);
            }
            cur = trie[cur].children[cv];
            trie[cur].cnt++;
            if (trie[cur].cnt >= K) {
                lengths.insert(i + 1);
            }
        }
    }
    void erase(const string& s) {
        int cur = 0;
        for (int i = 0; i < s.size(); ++i) {
            int cv = s[i] - 'a';
            cur = trie[cur].children[cv];
            trie[cur].cnt--;
            if (trie[cur].cnt == K - 1) {
                auto it = lengths.find(i + 1);
                lengths.erase(it);
            }
        }
    }
};
```

## bitwise trie data structure

The simplest for finding the largest xor sum subarray

```cpp
const int BITS = 30;
struct Node {
    int children[2];
    int last;
    void init() {
        memset(children, 0, sizeof(children));
        last = -1;
    }
};
struct Trie {
    vector<Node> trie;
    void init() {
        Node root;
        root.init();
        trie.emplace_back(root);
    }
    void add(int mask) {
        int cur = 0;
        for (int i = BITS - 1; i >= 0; i--) {
            int bit = (mask >> i) & 1;
            if (trie[cur].children[bit] == 0) {
                Node root;
                root.init();
                trie[cur].children[bit] = trie.size();
                trie.emplace_back(root);
            }
            cur = trie[cur].children[bit];
        }
    }
    int find(int val) {
        int cur = 0, ans = 0;
        for (int i = BITS - 1; i >= 0; i--) {
            int valBit = (val >> i) & 1;
            if (trie[cur].children[valBit ^ 1] != 0) { // best result with xor is always with opposite bit
                ans |= (1 << i);
                cur = trie[cur].children[valBit ^ 1];
            } else {
                cur = trie[cur].children[valBit];
            }
        }
        return ans;
    }
};
```

## 🔍 Purpose

This algorithm finds the **shortest subarray** in array `A` such that the **XOR of two elements** `A[i]` and `A[j]` within the subarray satisfies:  

---

## 🧠 Core Idea

- Use a **bitwise trie** (prefix tree) to store numbers from the array as binary strings.
- While iterating over the array:
  - Insert the current number into the trie with its index.
  - Query the trie to find the largest index `j` such that `A[i] ^ A[j] ≥ K`.
  - Track the minimum distance between such pairs to find the shortest subarray.

---

## ⚙️ Key Components

### 1. `Trie` Structure

- Each node has two children (`0` and `1`) and stores the **last index** of an inserted number on that path.
- Max bit length is 30 (`BITS = 30`) to support integers up to ~1e9.

### 2. `add(mask, idx)`

- Adds the binary representation of `mask` (an element of `A`) into the trie.
- Updates the `last` field to remember the latest index on that path.

### 3. `find(val, border)`

- Searches for the **largest index** `j` in the trie such that:  


```cpp
const int BITS = 30;
int N, K;
vector<int> A;

struct Node {
    int children[2];
    int last;
    void init() {
        memset(children, 0, sizeof(children));
        last = -1;
    }
};
struct Trie {
    vector<Node> trie;
    void init() {
        Node root;
        root.init();
        trie.emplace_back(root);
    }
    void add(int mask, int idx) {
        int cur = 0;
        trie[cur].last = max(trie[cur].last, idx);
        for (int i = BITS - 1; i >= 0; i--) {
            int bit = (mask >> i) & 1;
            if (trie[cur].children[bit] == 0) {
                Node root;
                root.init();
                trie[cur].children[bit] = trie.size();
                trie.emplace_back(root);
            }
            cur = trie[cur].children[bit];
            trie[cur].last = max(trie[cur].last, idx);
        }
    }
    int find(int val, int border) {
        int cur = 0, ans = -1;
        bool isMatching = true;
        for (int i = BITS - 1; i >= 0 && isMatching; i--) {
            int valBit = (val >> i) & 1;
            int borderBit = (border >> i) & 1;
            if (borderBit) {
                if (trie[cur].children[valBit ^ 1] != 0) {
                    cur = trie[cur].children[valBit ^ 1];
                } else {
                    isMatching = false;
                }
            } else {
                if (trie[cur].children[valBit ^ 1] != 0) {
                    ans = max(ans, trie[trie[cur].children[valBit ^ 1]].last); // pointers to next node on this path. 
                }
                if (trie[cur].children[valBit] != 0) {
                    cur = trie[cur].children[valBit];
                } else {
                    isMatching = false;
                }
            }
        }
        if (isMatching) {
            ans = max(ans, trie[cur].last);
        }
        return ans;
    }
};

void solve() {
    cin >> N >> K;
    A.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    Trie trie;
    trie.init();
    int ans = N + 1;
    for (int i = 0; i < N; i++) {
        trie.add(A[i], i);
        int j = trie.find(A[i], K);
        if (j != -1) ans = min(ans, i - j + 1);
    }
    if (ans == N + 1) {
        cout << -1 << endl;
        return;
    }
    cout << ans << endl;
}
```

## Prefix Tree for maximizing bitwise and storing the sorted indices in each node

prefix tree search for each range and determines what range of integers contain that prefix. This is useful for finding the maximum bitwise and of a range of integers, when you select a subset of integers from a range.

This prefix tree data structure allows you to find the maximum prefix with the constraint that the count of elements within a range have at least k elements with that prefix.  

By storing the index sorted at each node, you can binary search to figure out how many elements within the current range contain that prefix.  And only if it exceeds k do you pick the better prefix, else you have to explore all the current prefixes

```cpp
const int BITS = 18;
vector<int> arr;

// bit variant of prefix tree

int length(int l, int r) {
    return r - l;
}

// node structure for prefix tree
struct Node {
    int children[2];
    vector<int> indices;
    void init(vector<int>& arr) {
        memset(children, 0, sizeof(children));
        swap(arr, indices);
    }
};
struct PrefixTree {
    vector<Node> tree;
    void init(int n) {
        vector<int> nums(N);
        iota(nums.begin(), nums.end(), 0);
        Node root;
        root.init(nums);
        tree.push_back(root);
        build();
    }
    void build() {
        stack<int> stk, nstk;
        stk.push(0);
        for (int b = BITS; b >= 0; b--) {
            while (!stk.empty()) {
                int cur = stk.top();
                stk.pop();
                vector<int> zero, one;
                for (int idx : tree[cur].indices) {
                    if ((arr[idx] >> b) & 1) {
                        one.push_back(idx);
                    } else {
                        zero.push_back(idx);
                    }
                }
                if (!zero.empty()) {
                    Node root;
                    root.init(zero);
                    tree[cur].children[0] = tree.size();
                    nstk.push(tree.size());
                    tree.push_back(root);
                }
                if (!one.empty()) {
                    Node root;
                    root.init(one);
                    tree[cur].children[1] = tree.size();
                    nstk.push(tree.size());
                    tree.push_back(root);
                }
            }
            swap(nstk, stk);
        }
    }
    int search(int l, int r, int k) {
        vector<int> level, next_level;
        level.push_back(0);
        int res = 0;
        for (int b = BITS; b >= 0; b--) {
            next_level.clear();
            int cnt = 0;
            for (int cur : level) {
                int nxt = tree[cur].children[1];
                if (nxt) {
                    int s, e;
                    e = upper_bound(tree[nxt].indices.begin(), tree[nxt].indices.end(), r) - tree[nxt].indices.begin();
                    s = lower_bound(tree[nxt].indices.begin(), tree[nxt].indices.end(), l) - tree[nxt].indices.begin();
                    cnt += length(s, e);
                    next_level.push_back(nxt);
                }
                if (cnt >= k) break;
            }
            if (cnt < k) {
                for (int cur : level) {
                    int nxt = tree[cur].children[0];
                    if (nxt) {
                        next_level.push_back(nxt);
                    }
                }
            } else {
                res |= (1 << b);
            }
            swap(next_level, level);
        }
        return res;
    }
};

PrefixTree pretree;
pretree.init(N);

```