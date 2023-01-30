# LRU Cache 

This is probably fast lru cache in python.  slightly battle tested with just leetcode online judge

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
```