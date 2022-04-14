"""
QuickSelect algorithm that has worse case time complexity of O(n^2), but if you shuffle the input array it can on 
average have time complexity of O(n)

This can be used to select the kth smallest or kth largest element in an array
"""
def quickselect(arr, k):
  def partition(left, right):
      pval = arr[right]
      i = left
      for j in range(left, right):
          if arr[j] <= pval:
              arr[i], arr[j] = arr[j], arr[i]
              i += 1
      arr[i], arr[right] = arr[right], arr[i]
      return i
  def select(left, right):
      while left <= right:
          pindex = partition(left, right)
          if pindex == k - 1:
              return arr[pindex]
          elif pindex > k - 1:
              right = pindex - 1
          else:
              left = pindex + 1
      return -1
  return select(0,len(arr)-1)

"""
TODO: quickselect that uses random pivot index
"""