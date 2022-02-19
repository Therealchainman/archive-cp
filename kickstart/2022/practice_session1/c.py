import heapq
def h_index(n, citations):
  ans = []
  # TODO: Complete the function to get the H-Index scores after each paper
  cur = 0
  heap = []
  for cit in citations:
    heapq.heappush(heap,cit)
    while len(heap)>0 and heap[0]<=cur:
      heapq.heappop(heap)
    if len(heap)>cur:
      cur+=1
      ans.append(cur)
    else:
      ans.append(cur)
  return ans


if __name__ == '__main__':
  t = int(input())

  for test_case in range(1, t + 1):
    n = int(input())                      # The number of papers
    citations = map(int, input().split()) # The number of citations for each paper
    h_index_scores = h_index(n, citations)
    print("Case #" + str(test_case) + ": " + ' '.join(map(str, h_index_scores)))
