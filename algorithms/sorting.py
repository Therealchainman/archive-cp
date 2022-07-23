

"""
heap sort algorithm
average, best, worst case time complexity is O(nlogn)
the space complexity is O(1)
"""
# class HeapSort:
  


"""
merge sort algorithm for an array (list)
divide and conquer algorithm for stable sort of elements 
"""
class MergeSort:
    def __init__(self, arr):
        self.arr = arr
        
    def run(self, left, right):
        self.merge_sort(left, right)
        return self.arr
    
    def merge_sort(self, left, right):
        if right-left <= 1: return
        mid = (left+right)>>1
        self.merge_sort(left, mid)
        self.merge_sort(mid, right)
        self.merge(left,right,mid)
   
    def merge(self, left, right, mid):
        i, j = left, mid
        temp = []
        while i < mid and j < right:
            if self.arr[i] <= self.arr[j]:
                temp.append(self.arr[i])
                i += 1
            else:
                temp.append(self.arr[j])
                j+=1
        while i < mid:
            temp.append(self.arr[i])
            i += 1
        while j < right:
            temp.append(self.arr[j])
            j += 1
        for i in range(left, right):
            self.arr[i] = temp[i-left]