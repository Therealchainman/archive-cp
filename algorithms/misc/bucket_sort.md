


```py
from typing import List
def bucket_sort(nums: List[int]) -> List[int]:
    m = max(nums)
    bucket = [0] * (m + 1)
    for num in nums:
        bucket[num] += 1
    return bucket
```