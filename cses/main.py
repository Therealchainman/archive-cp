import os,sys
from io import BytesIO, IOBase
from typing import List
from collections import deque

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

def min_string_rotation(s: str) -> str:
    min_char = min(s)
    s_len = len(s)
    advance = lambda index: (index + 1)%s_len
    champions = deque()
    for i, ch in enumerate(s):
        if ch == min_char:
            champions.append(i)
    while len(champions) > 1:
        champion1 = champions.popleft()
        champion2 = champions.popleft()
        if champion2 < champion1:
            champion1, champion2 = champion2, champion1
        current_champion = champion1
        left_champion, right_champion = champion1, champion2
        for _ in range(champion2 - champion1):
            if s[left_champion] < s[right_champion]: break
            if s[left_champion] > s[right_champion]:
                current_champion = champion2
                break
            left_champion = advance(left_champion)
            right_champion = advance(right_champion)
        champions.append(current_champion)
    champion_index = champions.pop()
    return s[champion_index:] + s[:champion_index]

def main():
    s = input()
    return min_string_rotation(s)

if __name__ == '__main__':
    print(main())