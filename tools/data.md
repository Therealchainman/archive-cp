# DATA

## Creating a python list of random integers

```py
import random
low, high = 0, 10_000
nums = [random.randint(low, high) for _ in range(50_000)]
```