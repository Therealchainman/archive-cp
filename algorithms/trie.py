"""
A method to do trie
"""
from collections import deque, defaultdict
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
        """
        This is very special bfs for finding the count of number of matches for a word
        that is decrypted into a trie datastructure.  
        TODO: fix, and optimize this code.  
        """
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
"""
easier method with just a single class for TrieNode with defaultdict to represent the next triednoes
"""
class TrieNode:
    def __init__(self, name):
        self.children = defaultdict(TrieNode)
        self.name = name
        self.value = -1

class TrieNode:
    def __init__(self, index=-1):
        self.children = defaultdict(TrieNode)
        self.index = index

class TrieNode:
    def __init__(self, count_: int = 0):
        self.children = defaultdict(TrieNode)
        self.count = count_
        
    def __repr__(self) -> str:
        return f'count: {self.count}, children: {self.children}'

"""
This one is perfect for when matching a string to string of characters
such as does 'bad' exist in the trie datastructure
"""
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.isWord = False
    def __repr__(self):
        return f'is_word: {self.isWord}, children: {self.children}'