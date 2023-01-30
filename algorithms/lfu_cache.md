# LFU Cache

This is probably fast lfu cache in python.  slightly battle tested with just leetcode online judge

```py
class LRUCache:

    def __init__(self, capacity: int = math.inf):
        self.cache = OrderedDict()
        self.cap = capacity

    def get(self, key: int) -> int:
        if key not in self.cache: return -1
        self.put(key, self.cache[key])
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.cap:
            self.cache.popitem(last = False)

    def discard(self, key: int) -> None:
        if key not in self.cache: return
        self.cache.move_to_end(key)
        self.cache.popitem() # pops item from the end

    """
    returns key of lru item or -1 if lru cache is empty
    """
    def pop(self) -> int:
        try:
            return self.cache.popitem(last = False)[0] # pops item from start and returns just the key
        except:
            return -1

    def __repr__(self):
        return str(self.cache)
    
    def __len__(self):
        return len(self.cache)

    def __getitem__(self, key):
        return self.cache[key]

class LFUCache:

    def __init__(self, capacity: int):
        self.freq_key = dict() # points to the frequency of this key in lfu cache
        self.freq_lru_cache = defaultdict(LRUCache) # frequency cache construct of lru caches with infinite capacities
        self.min_freq_ptr = 0 # pointer to the least frequency with key or keys
        self.total_count = 0 # count of objects in lfu cache
        self.capacity = capacity # capacity of the lfu

    def get(self, key: int) -> int:
        if key not in self.freq_key: return -1
        freq = self.freq_key[key]
        val = self.freq_lru_cache[freq][key]
        self.put(key, val)
        return val

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0: return
        freq = self.freq_key.get(key, 0)
        if freq == 0:
            self.total_count += 1
        if self.total_count > self.capacity:
            lfu_key = self.freq_lru_cache[self.min_freq_ptr].pop()
            self.freq_key.pop(lfu_key)
            self.total_count -= 1
        self.freq_lru_cache[freq].discard(key)
        if freq == 0 or (freq == self.min_freq_ptr and len(self.freq_lru_cache[freq]) == 0):
            self.min_freq_ptr = freq + 1
        self.freq_lru_cache[freq + 1].put(key, value)
        self.freq_key[key] = freq + 1
```