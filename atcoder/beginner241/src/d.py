from bisect import bisect_left, bisect_right, insort_left 
from array import array

def sequence_query():

  Q = int(input())
  arr = array("i", [])

  queries = [list(map(int,input().split())) for _ in range(Q)]
  x_values = [queries[i][1] for i in range(Q)]
  x_values = set(x_values)
  x_values = sorted(list(x_values))
  to = {}
  fr = {}
  for i in range(len(x_values)):
    to[x_values[i]] = i
    fr[i] = x_values[i]
  for i in range(Q):
    query = queries[i]
    if query[0] == 1:
      x_index = to[query[1]]
      insort_left(arr, x_index) # nlogn
    elif query[0] == 2:
      x_index = to[query[1]]
      j = bisect_right(arr,x_index)
      k = query[2]
      if k > j:
        print(-1)
      else:
        print(fr[arr[j-k]])
    else:
      x_index = to[query[1]]
      j = bisect_left(arr,x_index)
      k = query[2]
      if k > len(arr) - j:
        print(-1)
      else:
        print(fr[arr[j+k-1]])

class FenwickTree:
    """
    Fenwick tree that is 1-index based, so must start all computations from 1 to n values
    """
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):

        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def __repr__(self):
        return f"array: {self.sums}"

def sequence_query_bit():
  Q = int(input())

  queries = [list(map(int,input().split())) for _ in range(Q)]
  x_values = [queries[i][1] for i in range(Q)]
  x_values.append(0)
  x_values = set(x_values)
  x_values = sorted(list(x_values))
  to = {}
  fr = {}
  for i, x in enumerate(x_values):
    to[x] = i
    fr[i] = x
  bit = FenwickTree(len(x_values))
  
  for query in queries:
    if query[0] == 1:
      x_index = to[query[1]]
      bit.update(x_index, 1)
    elif query[0] == 2:
      x, k = query[1:]
      x_index = to[x]
      fz = bit.query(x_index)
      if fz < k:
        print(-1)
      else:
        ok = 0
        ng = x_index + 1
        while ng-ok > 1:
          m = (ok+ng)//2
          if fz - bit.query(m-1) < k:
            ng = m
          else:
            ok = m
        print(fr[ok])
    else:
      x, k = query[1:]
      x_index = to[x]
      fz = bit.query(x_index)
      if bit.query(len(x_values)) - fz < k:
        print(-1)
      else:
        ok = len(x_values)
        ng = x_index - 1
        while ok - ng > 1:
          m = (ok+ng)//2
          if bit.query(m) - fz < k:
            ng = m
          else:
            ok = m
        print(fr[ok])
  
    


if __name__ == '__main__':
  # sequence_query()
  sequence_query_bit()