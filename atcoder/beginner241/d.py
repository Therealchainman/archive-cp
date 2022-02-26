from sortedcontainers import SortedList
from bisect import bisect_left, bisect_right

def ksmallest(pivot, k):
  i = A.bisect_right(pivot)
  if k < i: return -1
  return A[i-k]

def klargest(pivot, k):
  i = A.bisect_left(pivot)
  if k >= len(A)-i: return -1
  return A[i+k]


if __name__ == '__main__':
  Q = int(input())
  A = SortedList()  
  A.add(0)
  A.add(1000000000000000001)
  for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
      A.add(query[1])
    elif query[0] == 2:
      print(klargest(query[1],query[2]))
    else:
      print(ksmallest(query[1],query[2]))