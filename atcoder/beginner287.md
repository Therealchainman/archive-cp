# Atcoder Beginner Contest 287

## A. Majority

### Solution 1:  counter

```py
import os,sys
from io import BytesIO, IOBase
from collections import Counter

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

def main():
    n = int(input())
    counts = Counter()
    for _ in range(n):
        s = input()
        counts[s] += 1
    if counts['For'] > counts['Against']:
        return 'Yes'
    else:
        return 'No'

if __name__ == '__main__':
    print(main())
```

## B. Postal Card

### Solution 1:  string + set

```py
import os,sys
from io import BytesIO, IOBase
from collections import Counter

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

def main():
    n, m = map(int, input().split())
    res = 0
    arr = [None]*n
    t_arr = set()
    for i in range(n):
        arr[i] = input()
    for i in range(m):
        t_arr.add(input())
    for s in arr:
        if s[-3:] in t_arr:
            res += 1
    return res

if __name__ == '__main__':
    print(main())
```

## C. Path Graph?

### Solution 1: cycle detection in undirected graph + dfs

```py
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

def main():
    n, m = map(int, input().split())
    visited = [False]*n
    adj_list = defaultdict(list)
    for i in range(m):
        u, v = map(int, input().split())
        adj_list[u-1].append(v-1)
        adj_list[v-1].append(u-1)
    def is_cycle(node, parent_node):
        visited[node] = True
        for nei in adj_list[node]:
            if not visited[nei]:
                if is_cycle(nei, node):
                    return True
            elif nei != parent_node:
                return True
        return False
    cycle = is_cycle(0, None)
    return 'Yes' if not cycle and sum(visited) == n else 'No'

if __name__ == '__main__':
    print(main())
```

## D. Match or Not

### Solution 1:  prefix + suffix for boolean + bit manipulation + greedy

```py
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

def main():
    s = input()
    t = input()
    nt = len(t)
    ns = len(s)
    suffix_bool = [True]*(nt + 1)
    is_equal = lambda x, y: x == y or x == '?' or y == '?'
    for i in reversed(range(len(t))):
        suffix_bool[i] = (is_equal(t[i], s[ns - (nt - i)])) & suffix_bool[i + 1]
    result = [None]*(nt + 1)
    result[0] = 'Yes' if suffix_bool[0] else 'No'
    prefix_bool = True
    for i in range(nt):
        prefix_bool &= is_equal(t[i], s[i])
        res = suffix_bool[i + 1] & prefix_bool
        result[i + 1] = 'Yes' if res else 'No'
    return '\n'.join(result)


if __name__ == '__main__':
    print(main())
```

## E. Karuta

### Solution 1: max prefix in a trie data structure + add, remove, query operations for trie data structure

```py
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
        # FIND MAX PREFIX
        cur = root
        j = 0
        while j < len(word):
            ch = word[j]
            cur = cur[ch]
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
```

## F.

### Solution 1:

```py

```