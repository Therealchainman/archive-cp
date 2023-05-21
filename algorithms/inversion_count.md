# Inversion Count


```py
def inversionCount(self, arr, n):
    def merge(left, right, mid):
        nonlocal arr
        i, j = left, mid
        temp = []
        inv_count = 0
        while i < mid and j < right:
            if arr[i] <= arr[j]:
                temp.append(arr[i])
                i += 1
            else:
                inv_count += mid - i
                temp.append(arr[j])
                j += 1
        while i < mid:
            temp.append(arr[i])
            i += 1
        while j < right:
            temp.append(arr[j])
            j += 1
        for i in range(left, right):
            arr[i] = temp[i-left]
        return inv_count
        
    def merge_sort(left, right):
        if right - left <= 1: return 0
        mid = (left + right) >> 1
        inv_count = 0
        inv_count += merge_sort(left, mid)
        inv_count += merge_sort(mid, right)
        inv_count += merge(left, right, mid)
        return inv_count

    return merge_sort(0, n)
```