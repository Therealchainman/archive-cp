# TRIE


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