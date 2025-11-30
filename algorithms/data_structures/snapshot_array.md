# snapshot array

This implementation is memory optimize for sparse arrays, because if the values are initialized at 0 and are never changed, it doesn't do anything unless it has been set. it is optimized for setting values to 0 which is redundant as well.  

This snapshot array can be used when needing to store versions of an array, because need to access historical records of an array. 

```py
class SnapshotArray:

    def __init__(self, length: int):
        self.arr = {}
        self.snapshot_arr = defaultdict(list)
        self.version = 0

    # O(1)
    def set(self, index: int, val: int) -> None:
        if self.snapshot_arr[index] and self.snapshot_arr[index][-1][0] == self.version: 
            self.snapshot_arr[index][-1] = (self.version, val)
        elif self.snapshot_arr[index] and self.snapshot_arr[index][-1][1] == val: 
            return
        elif val != 0 or self.snapshot_arr[index]:
            self.snapshot_arr[index].append((self.version, val))

    # O(1)
    def snap(self) -> int:
        self.version += 1
        return self.version - 1

    # O(log(n))
    def get(self, index: int, snap_id: int) -> int:
        if not self.snapshot_arr[index] or self.snapshot_arr[index][0][0] > snap_id: return 0
        i = bisect.bisect_right(self.snapshot_arr[index], (snap_id, math.inf)) - 1
        return self.snapshot_arr[index][i][1]
```