from collections import deque
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