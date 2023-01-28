import os,sys
from io import BytesIO, IOBase
from collections import Counter, defaultdict
sys.setrecursionlimit(10**6)

# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

class TrieNode(defaultdict):
    def __init__(self):
        super().__init__(TrieNode)
        self.prefix_count = 0 # how many words have this prefix

    def __repr__(self) -> str:
        return f'is_word: {self.is_word} prefix_count: {self.prefix_count}, children: {self.keys()}'

def main():
    n = int(input())
    words = [input() for _ in range(n)]
    root = TrieNode()
    for word in words:
        cur = root
        for ch in word:
            cur = cur[ch]
            cur.prefix_count += 1
    result = [0]*n
    for i, word in enumerate(words):
        # REMOVE CURRENT WORD SO DON'T MATCH WITH ITSELF
        cur = root
        for ch in word:
            cur = cur[ch]
            cur.prefix_count -= 1
        cur = root
        # print('word', word)
        j = 0
        while j < len(word):
            ch = word[j]
            cur = cur[ch]
            # print('j', j, 'cur_count', cur.prefix_count)
            if cur.prefix_count == 0:
                break
            j += 1
        result[i] = j
        # ADD CURRENT WORD BACK
        cur = root
        for ch in word:
            cur = cur[ch]
            cur.prefix_count += 1



    return '\n'.join(map(str, result))


if __name__ == '__main__':
    print(main())