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

```cpp
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

Trie trie;
trie.init();
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