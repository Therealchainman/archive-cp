# Advent of Code 2016

## Day 1: 

### Solution 1:  

```py

```

## Day 4: 

### Solution 1:  string, rotation, map

```py
def unicode(ch):
    return ord(ch) - ord("a")
def decode(c):
    return chr(c + ord("a"))
def rotate(ch, v):
    if ch == "-": return " "
    return decode((unicode(ch) + v) % 26)
def calculate(filename: str):
    with open(filename, "r") as f:
        data = f.read().splitlines()
        p2 = r"\d+"
        for row in data:
            match = re.search(p2, row)
            val = int(match.group())
            decoded_string = "".join(map(lambda x: rotate(x, val), row[:match.start()]))
            print(decoded_string, val)
calculate("big.txt")
```

## Day 5:

### Solution 1:  md5 hash, hash, string, hexadecimal

```py
class GeneratePassword:
    def hash(self, s):
        return hashlib.md5(s.encode("utf-8")).hexdigest()
    def calculate(self, filename):
        with open(filename, "r") as f:
            data = f.read()
            ans = [-1] * 8
            rem = 8
            i = 0
            while rem > 0:
                h = self.hash(data + str(i))
                if h.startswith("00000"):
                    pos = int(h[5], 16)
                    if pos < 8 and ans[pos] == -1:
                        ans[pos] = h[6]
                        rem -= 1
                i += 1
                if i % 10_000_000 == 0: print(i)
            print("".join(ans))

GeneratePassword().calculate("big.txt")
```

## Day 6:

### Solution 1:  counts

```py
class Decode:
    def calculate(self, filename):
        with open(filename, "r") as f:
            data = f.read().splitlines()
            N = len(data[0])
            row_counts = [Counter() for _ in range(N)]
            for row in data:
                for i, ch in enumerate(row):
                    row_counts[i][ch] += 1
            ans = []
            for counts in row_counts:
                ans.append(counts.most_common()[-1][0])
            print("".join(ans))

Decode().calculate("big.txt")
```

## Day 7:

### Solution 1: 

```py

```

## Day 8:

### Solution 1: 

```py

```

## Day 9:

### Solution 1: 

```py

```

## Day 15 

### Solution 1:  Modular arithmetic, arrays

```py
disc = compile("Disc #{:d} has {:d} positions; at time=0, it is at position {:d}.")
with open('big.txt', 'r') as f:
    data = f.read().splitlines()
    targets = []
    positions = []
    pos = []
    for line in data:
        d, p, cur = disc.parse(line).fixed
        positions.append(p)
        pos.append(cur)
        targets.append((p - d) % p)
    positions.append(11)
    pos.append(0)
    targets.append((11 - 7) % 11)
    n = len(targets)
    res = 0
    while True:
        for i in range(n):
            pos[i] = (pos[i] + 1) % positions[i]
        res += 1
        if all(pos[i] == targets[i] for i in range(n)):
            break
    print(res)
```

## Day 16

### Solution 1:  list

```py
with open("big.txt", "r") as f:
    data = f.read()
    # n = 272 # part 1
    n = 35_651_584
    data = list(map(int, data))
    while len(data) <= n:
        nxt = [c ^ 1 for c in reversed(data)]
        data.append(0)
        data.extend(nxt)
    data = data[:n]
    while len(data) % 2 == 0:
        data = [1 if data[i] == data[i + 1] else 0 for i in range(0, len(data), 2)]
    print("".join(map(str, data)))
```

geometric series summation and logn time with bit manipulation and bit scan forward algorithm to find least significant bit.

```py
def bsf(n):
    pos = 0
    while n % 2 == 0:
        pos += 1
        n >>= 1
    return pos
with open("big.txt", "r") as f:
    data = f.read()
    # n = 272 # part 1
    n = 35_651_584
    data = list(map(int, data))
    while len(data) <= n:
        nxt = [c ^ 1 for c in reversed(data)]
        data.append(0)
        data.extend(nxt)
    data = data[:n]
    m = bsf(n)
    delta = 1
    for _ in range(m):
        for i in range(0, len(data), 2 * delta):
            data[i] = 1 if data[i] == data[i + delta] else 0
        delta <<= 1
    print("".join(map(str, [data[i] for i in range(0, n, delta)])))
```

## Day 17

### Solution 1:

```py

```

## Day 16

### Solution 1:

```py

```

## Day 16

### Solution 1:

```py

```

## Day 16

### Solution 1:

```py

```