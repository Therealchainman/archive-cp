import os,sys
from io import BytesIO, IOBase

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

from math import inf
def main():
    num_cities, num_roads, num_queries = map(int,input().split())
    dist = [[inf]*(num_cities+1) for _ in range(num_cities+1)]
    for i in range(num_cities+1):
        dist[i][i] = 0
    for _ in range(num_roads):
        city1, city2, length = map(int,input().split())
        dist[city1][city2] = dist[city2][city1] = length
    for k in range(1, num_cities+1):
        for i in range(1,num_cities+1):
            if dist[i][k] == inf: continue
            for j in range(1,num_cities+1):
                if dist[k][j] == inf: continue
                dist[i][j] = min(dist[i][j], dist[i][k]+dist[j][k])
    return "\n".join(map(str, (dist[city1][city2] if dist[city1][city2] != inf else -1 \
    for city1, city2 in [map(int,input().split()) for _ in range(num_queries)])))

if __name__ == '__main__':
    print(main())