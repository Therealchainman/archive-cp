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
        return f'is_word: {self.is_word} prefix_count: {self.prefix_count}, children: {self.keys()}'s
```