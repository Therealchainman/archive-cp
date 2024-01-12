# Advent of Code 2023

## Day 1

### Solution 1:  two pointers

```py
def leftmost_digit(line):
    return next(dropwhile(lambda x: not x.isdigit(), line))
def rightmost_digit(line):
    return next(dropwhile(lambda x: not x.isdigit(), reversed(line)))
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    p1 = 0
    for line in data:
        x, y = leftmost_digit(line), rightmost_digit(line)
        p1 += int(x + y)
    print("part 1:", p1)
```

### Solution 2:  regex search, and greedy search

```py
def leftmost_digit(line):
    return re.search(r"\d", line).group(0)
def rightmost_digit(line):
    return re.search(r".*\d", line).group(0)[-1]
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    p1 = 0
    for line in data:
        x, y = leftmost_digit(line), rightmost_digit(line)
        p1 += int(x + y)
    print("part 1:", p1)
```

### Solution 3:  regex search for everything, dictionary to convert

```py
pattern = r"(\d|one|two|three|four|five|six|seven|eight|nine)"
str_to_digits = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
def leftmost_digit(line):
    return re.search(pattern, line).group(1)
def rightmost_digit(line):
    return re.search(".*" + pattern, line).group(1)
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    ans = 0
    for line in data:
        x, y = leftmost_digit(line), rightmost_digit(line)
        x, y = str_to_digits.get(x, x), str_to_digits.get(y, y)
        ans += int(x + y)
    print("part 2:", ans)
```

## Day 2

### Solution 1:  regex search, regex findall, regex groups, dictionaries, any

```py
upper_limits = {"red": 12, "green": 13, "blue": 14}
def find_minimal(parts):
    desired_counts = {"red": 0, "green": 0, "blue": 0}
    for part in parts:
        part_data = re.findall(pattern, part)
        for x, y in part_data:
            desired_counts[y] = max(desired_counts[y], int(x))
    return desired_counts
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    p1 = p2 = 0
    game_pat = "Game (\d+): (.*)"
    pattern = "(\d+) (\w+)"
    for line in data:
        game_id = int(re.search(game_pat, line).group(1))
        game_data = re.search(game_pat, line).group(2)
        parts = game_data.split("; ")
        counts = find_minimal(parts)
        if all(v <= upper_limits[k] for k, v in counts.items()): p1 += game_id
        p2 += math.prod(counts.values())
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 3

### Solution 1:  regex finditer, dictionary of list, grid

```py
def symbol(c1, c2, r, v):
    ans = 0
    for c in range(c1 - 1, c2 + 1):
        for dr in range(-1, 2):
            if not in_bounds(r + dr, c) or grid[r + dr][c] in ".0123456789": continue
            ans += 1
            if grid[r + dr][c] == "*":
                gears[(r + dr, c)].append(v)
    return ans
with open("big.txt", "r") as f:
    grid = f.read().splitlines()
    R, C = len(grid), len(grid[0])
    in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
    p1 = 0
    pat = "\d+"
    gears = defaultdict(list)
    for c, row in enumerate(grid):
        for match_ in re.finditer(pat, row):
            val = int(match_.group())
            count = symbol(match_.start(), match_.end(), c, val)
            if count > 0: p1 += val
    p2 = sum(math.prod(lst) for lst in gears.values() if len(lst) == 2)
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 4

### Solution 1:  dynamic programming, regex findall

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    p1 = 0
    card_pat = ".*\d+:(.*)"
    num_pat = "\d+"
    copies = [1] * len(data)
    for i, line in enumerate(data):
        numbers = re.search(card_pat, line).group(1)
        winning_numbers, my_numbers = numbers.split("|")
        cnt = 0
        for num in re.findall(num_pat, my_numbers):
            if f" {num} " in winning_numbers: cnt += 1
        if cnt > 0:
            p1 += pow(2, cnt - 1)
        for j in range(1, cnt + 1):
            copies[i + j] += copies[i]
    print("part 1:", p1)
    print("part 2:", sum(copies))
```

## Day 5

### Solution 1:  ranges, intersection of 1d ranges

```py
def apply_ranges(ranges, maps):
    ans = []
    for dest, src, len_ in maps:
        start, end = src, src + len_
        new_ranges = []
        for s, e in ranges:
            before = (s, min(e, start))
            mid = (max(s, start), min(e, end))
            after = (max(s, end), e)
            if before[1] - before[0] > 0: new_ranges.append(before)
            if mid[1] - mid[0] > 0: ans.append((dest + mid[0] - start, dest + mid[1] - start))
            if after[1] - after[0] > 0: new_ranges.append(after)
        ranges = new_ranges
    ans.extend(new_ranges)
    return ans
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    p1 = p2 = math.inf
    ptr = -1
    maps = [[] for _ in range(7)]
    for line in data[2:]:
        if line == "": continue
        if ":" in line: ptr += 1; continue
        dest, src, len_ = map(int, line.split())
        maps[ptr].append((dest, src, len_))
    seed_pat = "seeds: (.*)"
    seeds = list(map(int, re.search(seed_pat, data[0]).group(1).split()))
    for i in range(len(seeds)):
        if i & 1:
            ranges = [(seeds[ i - 1], seeds[i - 1] + seeds[i])]
            for j in range(7):
                ranges = apply_ranges(ranges, maps[j])
            p2 = min(p2, sorted(ranges)[0][0])
        ranges = [(seeds[i], seeds[i] + 1)]
        for j in range(7):
            ranges = apply_ranges(ranges, maps[j])
        p1 = min(p1, sorted(ranges)[0][0])
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 6

### Solution 1:  binary search, quadratic function, peak

```py
def win(d, t):
    peak = t // 2
    start = bisect.bisect_left(range(peak + 1), d, key = lambda x: x * (t - x))
    end = bisect.bisect_right(range(peak, t + 1), False, key = lambda x : x * (t - x) <= d) + peak
    return end - start
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    times = list(map(int, re.findall("\d+", data[0])))
    distances = list(map(int, re.findall("\d+", data[1])))
    n = len(times)
    p1 = 1
    for d, t in zip(distances, times):
        p1 *= win(d, t)
    t = int("".join(map(str, times)))
    d = int("".join(map(str, distances)))
    p2 = win(d, t)
    print("part 1:", p1)
    print("part 2:", p2)
```

### Solution 2:  Quadratic Solver

```py
def win(d, t):
    discriminant = t * t - 4 * d
    assert discriminant > 0
    sqrt_discriminant = math.sqrt(discriminant)
    min_x = math.floor((t - sqrt_discriminant) / 2) + 1
    max_x = math.ceil((t + sqrt_discriminant) / 2) - 1
    return max_x - min_x + 1
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    times = list(map(int, re.findall("\d+", data[0])))
    distances = list(map(int, re.findall("\d+", data[1])))
    n = len(times)
    p1 = 1
    for d, t in zip(distances, times):
        p1 *= win(d, t)
    t = int("".join(map(str, times)))
    d = int("".join(map(str, distances)))
    p2 = win(d, t)
    print("part 1:", p1)
    print("part 2:", p2)
```

### Solution 3:  Use symmetry of parabola

```py
def win(d, t):
    discriminant = t * t - 4 * d
    assert discriminant > 0
    sqrt_discriminant = math.sqrt(discriminant)
    min_x = math.floor((t - sqrt_discriminant) / 2) + 1
    return int(2 * (t / 2 - min_x) + 1)
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    times = list(map(int, re.findall("\d+", data[0])))
    distances = list(map(int, re.findall("\d+", data[1])))
    n = len(times)
    p1 = 1
    for d, t in zip(distances, times):
        p1 *= win(d, t)
    t = int("".join(map(str, times)))
    d = int("".join(map(str, distances)))
    p2 = win(d, t)
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 7

### Solution 1: custom sorting, counts of cards

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    hands = []
    for line in data:
        hand, bid = line.split()
        bid = int(bid)
        hands.append((hand, bid))
    cards = "**23456789TJQKA"
    def hand_rank(hand):
        return sum(hand.count(card) for card in hand), tuple(cards.index(card) for card in hand)
    hands.sort(key = lambda x: hand_rank(x[0]))
    p1 = sum(rank * bid for rank, (_, bid) in enumerate(hands, start = 1))
    print("part 1:", p1)
```

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    hands = []
    for line in data:
        hand, bid = line.split()
        bid = int(bid)
        hands.append((hand, bid))
    cards = "*J23456789TQKA"
    def hand_rank(hand):
        if hand == "JJJJJ": return (25, (1, 1, 1, 1, 1))
        mode = max(filter(lambda x: x != "J", hand), key = hand.count)
        new_hand = hand.replace("J", mode)
        return sum(new_hand.count(card) for card in new_hand), tuple(cards.index(card) for card in hand)
    hands.sort(key = lambda x: hand_rank(x[0]))
    p2 = sum(rank * bid for rank, (_, bid) in enumerate(hands, start = 1))
    print("part 2:", p2)
    

```

## Day 8

### Solution 1:  modular arithmetic, network, cycle detection, least common multiple

```py
with open('big.txt', 'r') as f:
    data = f.read().splitlines()
    n = len(data)
    p1, p2 = 0, 1
    instructions = data[0]
    n = len(instructions)
    network = {}
    for line in data[2:]:
        src, L, R = re.findall("\w+", line)
        network[src] = (L, R)
    for start in network.keys():
        if not start.endswith("A"): continue
        node = start
        cnt = 0
        while not node.endswith("Z"):
            node = network[node][0 if instructions[cnt % n] == "L" else 1]
            cnt += 1
        assert cnt % n == 0
        assert network[start][0 if instructions[0] == "L" else 1] == network[node][0 if instructions[-1] == "L" else 1]
        p2 = math.lcm(p2, cnt)
        if start == "AAA":
            p1 = cnt
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 9

### Solution 1:  dynamic programming, sum of integers, arithmetic series

```py
def process(row):
    last = [row[-1]]
    while True:
        next_row = [0] * (len(row) - 1)
        for i in range(1, len(row)):
            next_row[i - 1] = row[i] - row[i - 1]
        row = next_row
        if all(x == 0 for x in row): break
        last.append(row[-1])
    return sum(last)
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    p1 = p2 = 0
    for line in data:
        row = list(map(int, line.split()))
        p1 += process(row)
        p2 += process(row[::-1])
    print("part 1:", p1)
    print("part 2:", p2)
```
  
## Day 10

### Solution 1:  point in polygon, ray casting, rectilinear polygon

```py
U = (-1, 0)
D = (1, 0)
L = (0, -1)
R = (0, 1)

pipes = {
    "|": (U, D),
    "F": (D, R),
    "J": (U, L),
    "7": (D, L),
    "L": (U, R),
    "-": (L, R),
    ".": ()
}
def move(r, c, pr, pc, grid):
    for dr, dc in pipes[grid[r][c]]:
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc) or (nr, nc) == (pr, pc): continue
        return nr, nc
    return -1, -1
def point_in_polygon(grid, vis):
    N = len(grid)
    inside = 0
    for r in range(N):
        up = down = 0
        for c in range(N):
            if not vis[r][c] and up and down:
                inside += 1
            if vis[r][c]:
                for heading in pipes[grid[r][c]]:
                    if heading == U: up ^= 1
                    elif heading == D: down ^= 1
    return inside
with open("big.txt", "r") as f:
    grid = [list(row) for row in f.read().splitlines()]
    # assert it is a square grid
    assert len(grid) == len(grid[0])
    N = len(grid)
    in_bounds = lambda r, c: 0 <= r < N and 0 <= c < N
    neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    sr = sc = None
    for r, c in product(range(N), repeat = 2):
        if grid[r][c] == "S": sr, sc = r, c; break
    assert sr is not None
    start = set("|-FJL7")
    if grid[sr - 1][sc] not in "|F7":
        start -= set("|LJ")
    if grid[sr + 1][sc] not in "|LJ":
        start -= set("|F7")
    if grid[sr][sc - 1] not in "-FL":
        start -= set("-J7")
    if grid[sr][sc + 1] not in "-J7":
        start -= set("-FL")
    assert len(start) == 1
    grid[sr][sc] = start.pop()
    r, c = sr, sc
    vis = [[0] * N for _ in range(N)]
    vis[r][c] = 1
    pr, pc = sr, sc
    r, c = move(r, c, pr, pc, grid)
    while (r, c) != (sr, sc):
        vis[r][c] = 1
        nr, nc = move(r, c, pr, pc, grid)
        pr, pc = r, c
        r, c = nr, nc
    p2 = point_in_polygon(grid, vis)
    print("part 1:", sum(sum(row) for row in vis) // 2)
    print("part 2:", p2)
```

### Solution 2:  Pick's theorem, shoelace formula

```py
U = (-1, 0)
D = (1, 0)
L = (0, -1)
R = (0, 1)

pipes = {
    "|": (U, D),
    "F": (D, R),
    "J": (U, L),
    "7": (D, L),
    "L": (U, R),
    "-": (L, R),
    ".": ()
}
def move(r, c, pr, pc, grid):
    for dr, dc in pipes[grid[r][c]]:
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc) or (nr, nc) == (pr, pc): continue
        return nr, nc
    return -1, -1
def shoelace(vertices):
    double_area = 0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        double_area += x1 * y2 - x2 * y1
    double_area = abs(double_area)
    return double_area // 2
# find number of interior points
def picks_theorem(A, b):
    return A - b // 2 + 1
with open("big.txt", "r") as f:
    grid = [list(row) for row in f.read().splitlines()]
    # assert it is a square grid
    assert len(grid) == len(grid[0])
    N = len(grid)
    in_bounds = lambda r, c: 0 <= r < N and 0 <= c < N
    neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    sr = sc = None
    for r, c in product(range(N), repeat = 2):
        if grid[r][c] == "S": sr, sc = r, c; break
    assert sr is not None
    start = set("|-FJL7")
    if grid[sr - 1][sc] not in "|F7":
        start -= set("|LJ")
    if grid[sr + 1][sc] not in "|LJ":
        start -= set("|F7")
    if grid[sr][sc - 1] not in "-FL":
        start -= set("-J7")
    if grid[sr][sc + 1] not in "-J7":
        start -= set("-FL")
    assert len(start) == 1
    grid[sr][sc] = start.pop()
    r, c = sr, sc
    vertices = [(sr, sc)]
    pr, pc = sr, sc
    r, c = move(r, c, pr, pc, grid)
    while (r, c) != (sr, sc):
        vertices.append((r, c))
        nr, nc = move(r, c, pr, pc, grid)
        pr, pc = r, c
        r, c = nr, nc
    p2 = picks_theorem(shoelace(vertices), len(vertices))
    print("part 1:", len(vertices) // 2)
    print("part 2:", p2)
```

### Solution 3:  matplotlib Path object, creates a path from a set of integer coordinates, and then you can determine how many points are inside that enclosed path

```py
U = (-1, 0)
D = (1, 0)
L = (0, -1)
R = (0, 1)

pipes = {
    "|": (U, D),
    "F": (D, R),
    "J": (U, L),
    "7": (D, L),
    "L": (U, R),
    "-": (L, R),
    ".": ()
}
def move(r, c, pr, pc, grid):
    for dr, dc in pipes[grid[r][c]]:
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc) or (nr, nc) == (pr, pc): continue
        return nr, nc
    return -1, -1
def path(vertices):
    res = 0
    p = Path(vertices)
    boundary = set(vertices)
    for r, c in product(range(N), repeat = 2):
        if (r, c) in boundary: continue
        if p.contains_point((r, c)): res += 1
    return res
with open("big.txt", "r") as f:
    grid = [list(row) for row in f.read().splitlines()]
    # assert it is a square grid
    assert len(grid) == len(grid[0])
    N = len(grid)
    in_bounds = lambda r, c: 0 <= r < N and 0 <= c < N
    neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    sr = sc = None
    for r, c in product(range(N), repeat = 2):
        if grid[r][c] == "S": sr, sc = r, c; break
    assert sr is not None
    start = set("|-FJL7")
    if grid[sr - 1][sc] not in "|F7":
        start -= set("|LJ")
    if grid[sr + 1][sc] not in "|LJ":
        start -= set("|F7")
    if grid[sr][sc - 1] not in "-FL":
        start -= set("-J7")
    if grid[sr][sc + 1] not in "-J7":
        start -= set("-FL")
    assert len(start) == 1
    grid[sr][sc] = start.pop()
    r, c = sr, sc
    vertices = [(sr, sc)]
    pr, pc = sr, sc
    r, c = move(r, c, pr, pc, grid)
    while (r, c) != (sr, sc):
        vertices.append((r, c))
        nr, nc = move(r, c, pr, pc, grid)
        pr, pc = r, c
        r, c = nr, nc
    p2 = path(vertices)
    print("part 1:", len(vertices) // 2)
    print("part 2:", p2)
```

## Day 11

### Solution 1:  manhattand distance, 1d prefix sum for row and column, multiplier for empty rows or columns

```py
def solve(mult):
    ans = 0
    row_sums = [0] * (N + 1)
    col_sums = [0] * (N + 1)
    for c in range(N):
        col_sums[c + 1] = col_sums[c] + (mult if cols[c] == 0 else 1) * 1
    for r in range(N):
        row_sums[r + 1] = row_sums[r] + (mult if rows[r] == 0 else 1) * 1
    for i in range(len(galaxies)):
        for j in range(i):
            r1, c1 = galaxies[i]
            r2, c2 = galaxies[j]
            if r1 > r2:
                r1, r2 = r2, r1
            if c1 > c2:
                c1, c2 = c2, c1
            r_dist = row_sums[r2] - row_sums[r1]
            c_dist = col_sums[c2] - col_sums[c1]
            ans += r_dist + c_dist
    return ans
with open("big.txt", "r") as f:
    grid = [list(row) for row in f.read().splitlines()]
    # assert it is a square grid
    assert len(grid) == len(grid[0])
    N = len(grid)
    rows = [0] * N
    cols = [0] * N
    galaxies = []
    for r, c in product(range(N), repeat = 2):
        if grid[r][c] == "#":
            rows[r] += 1
            cols[c] += 1
            galaxies.append((r, c))
    p1, p2 = solve(2), solve(1_000_000)
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 12

### Solution 1:  dynamic programming with bags

```py
def solve(elements, groups):
    # state = (current group size, current group index, last character)
    bags = Counter()
    if elements[0] == "?" or elements[0] == "#":
        bags[(1, 0)] += 1
    if elements[0] == "?" or elements[0] == ".":
        bags[(0, 0)] += 1
    n = len(elements)
    m = len(groups)
    for i in range(1, n):
        new_bags = Counter()
        for (cur, j), cnt in bags.items():
            if elements[i] == ".":
                if cur == 0:
                    new_bags[(0, j)] += cnt
                elif cur == groups[j]:
                    new_bags[(0, j + 1)] += cnt
            elif elements[i] == "#":
                if j < m and cur < groups[j]:
                    new_bags[(cur + 1, j)] += cnt
            else:
                if cur == 0:
                    new_bags[(0, j)] += cnt
                    if j < m:
                        new_bags[(1, j)] += cnt
                elif cur == groups[j]:
                    new_bags[(0, j + 1)] += cnt
                elif cur > 0:
                    new_bags[(cur + 1, j)] += cnt
        bags = new_bags
    return sum(cnt for (_, j), cnt in bags.items() if j == m)
def parts(sz):
    with open("big.txt", "r") as f:
        data = f.read().splitlines()
        res = 0
        for line in data:
            first, second = line.split()
            start = list(map(int, second.split(",")))
            elements = "?".join([first for _ in range(sz)]) + "."
            groups = []
            for _ in range(sz):
                groups.extend(start)
            res += solve(elements, groups)
        return res
print("part 1:", parts(1))
print("part 2:", parts(5))
```

## Day 13

### Solution 1:  all, any, reverse, zip, find line of reflections in grid

```py
def solve(grid):
    R, C = len(grid), len(grid[0])
    def horizontal_lor(r):
        return all(row1 == row2 for row1, row2 in zip(grid[r:], reversed(grid[:r])))
    def vertical_lor(c):
        for row in grid:
            if any(col1 != col2 for col1, col2 in zip(row[c:], reversed(row[:c]))): return False
        return True
    ans = []
    for r in range(1, R):
        if horizontal_lor(r): ans.append(100 * r)
    for c in range(1, C):
        if vertical_lor(c): ans.append(c)
    return ans
def smudge(grid):
    R, C = len(grid), len(grid[0])
    original_ref = solve(grid)[0]
    for r, c in product(range(R), range(C)):
        tmp = grid[r][c]
        grid[r][c] = "." if tmp == "#" else "#"
        refs = solve(grid)
        grid[r][c] = tmp
        if not refs: continue
        for ref in refs:
            if ref != original_ref: return ref
    return None
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    data.append("")
    grid = []
    p1 = p2 = 0
    for row in data:
        if row == "":
            p1 += solve(grid)[0]
            p2 += smudge(grid)
            grid = []
        else:
            grid.append(list(row))
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 14

### Solution 1:  simulation

```py
def north():
    rocks = []
    for r, c in product(range(N), repeat = 2):
        if grid[r][c] == "O": rocks.append((r, c)); grid[r][c] = "."
    for r, c in rocks:
        while r > 0 and grid[r - 1][c] == ".": r -= 1
        grid[r][c] = "O"
with open("big.txt", "r") as f:
    grid = [list(row) for row in f.read().splitlines()]
    assert len(grid) == len(grid[0])
    N = len(grid)
    north()
    p1 = sum(N - r for r, c in product(range(N), repeat = 2) if grid[r][c] == "O")
    print("part 1:", p1)
```

### Solution 2:  simulation, cycle detection, functional graph

```py
def north():
    rocks = []
    for r, c in product(range(N), repeat = 2):
        if grid[r][c] == "O": rocks.append((r, c)); grid[r][c] = "."
    for r, c in rocks:
        while r > 0 and grid[r - 1][c] == ".": r -= 1
        grid[r][c] = "O"
def south():
    rocks = []
    for r, c in product(reversed(range(N)), repeat = 2):
        if grid[r][c] == "O": rocks.append((r, c)); grid[r][c] = "."
    for r, c in rocks:
        while r + 1 < N and grid[r + 1][c] == ".": r += 1
        grid[r][c] = "O"
def east():
    rocks = []
    for c, r in product(reversed(range(N)), repeat = 2):
        if grid[r][c] == "O": rocks.append((r, c)); grid[r][c] = "."
    for r, c in rocks:
        while c + 1 < N and grid[r][c + 1] == ".": c += 1
        grid[r][c] = "O"
def west():
    rocks = []
    for r, c in product(range(N), repeat = 2):
        if grid[r][c] == "O": rocks.append((r, c)); grid[r][c] = "."
    for r, c in rocks:
        while c > 0 and grid[r][c - 1] == ".": c -= 1
        grid[r][c] = "O"
def transformations():
    north()
    west()
    south()
    east()
    return [(r, c) for r, c in product(range(N), repeat = 2) if grid[r][c] == "O"]
with open("big.txt", "r") as f:
    grid = [list(row) for row in f.read().splitlines()]
    assert len(grid) == len(grid[0])
    N = len(grid)
    iterations = 1_000_000_000
    cnt = 0
    last = {tuple([(r, c) for r, c in product(range(N), repeat = 2) if grid[r][c] == "O"]): 0}
    while True:
        cnt += 1
        hash_ = tuple(transformations())
        if hash_ in last: cycle_len = cnt - last[hash_]; crit_point = last[hash_]; break
        last[hash_] = cnt
    print("cnt", cnt, "length of cycle", cycle_len)
    iterations -= crit_point
    i = iterations % cycle_len
    for node, index in last.items():
        if index == crit_point: solution = node; break
    p2 = sum(N - r for r, _ in solution)
    print("part 2:", p2)
```

## Day 15

### Solution 1:  hashmap, class, python dictionaries are ordered

```py
def hash_func(s):
    mod = 256
    h = 0
    for ch in s:
        h = ((ord(ch) + h) * 17) % mod
    return h
remove = parse.compile("{}-")
assign = parse.compile("{}={:d}")
class Box:
    def __init__(self, index):
        self.focals = {}
        self.index = index
    def add(self, label, focal_length):
        self.focals[label] = focal_length
    def remove(self, label):
        if label in self.focals:
            del self.focals[label]
    def evaluate(self):
        return self.index * sum(i * fl for i, fl in enumerate(self.focals.values(), start = 1))
with open("big.txt", "r") as f:
    data = f.read()
    p1 = p2 = 0
    boxes = [Box(i + 1) for i in range(256)]
    for s in data.split(","):
        p1 += hash_func(s)
        focal_length = None
        if assign.parse(s):
            label, focal_length = assign.parse(s).fixed
        else:
            label = remove.parse(s).fixed[0]
        hash_ = hash_func(label)
        if focal_length is None:
            boxes[hash_].remove(label)
        else:
            boxes[hash_].add(label, focal_length)
    p2 = sum(box.evaluate() for box in boxes)
    print("part 1:", p1)
    print("part 2:", p2)
```

### Solution 2:  hashmap, dictionaries

```py
def hash_func(s):
    mod = 256
    h = 0
    for ch in s:
        h = ((ord(ch) + h) * 17) % mod
    return h
remove = parse.compile("{}-")
assign = parse.compile("{}={:d}")
with open("big.txt", "r") as f:
    data = f.read()
    p1 = p2 = 0
    boxes = [{} for i in range(256)]
    for s in data.split(","):
        p1 += hash_func(s)
        focal_length = None
        if assign.parse(s):
            label, focal_length = assign.parse(s).fixed
        else:
            label = remove.parse(s).fixed[0]
        hash_ = hash_func(label)
        if focal_length is None:
            if label in boxes[hash_]: del boxes[hash_][label]
        else:
            boxes[hash_][label] = focal_length
    p2 = sum((i + 1) * sum(j * fl for j, fl in enumerate(boxes[i].values(), start = 1)) for i in range(256))
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 16

### Solution 1:  flood fill, bfs

```py
E = (0, 1)
W = (0, -1)
N = (-1, 0)
S = (1, 0)
with open("big.txt", "r") as f:
    grid = [list(row) for row in f.read().splitlines()]
    assert len(grid) == len(grid[0])
    n = len(grid)
    columns = {
        0: E,
        n - 1: W,
    }
    rows = {
        0: S,
        n - 1: N
    }
    in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
    def get(r, c, dr, dc):
        res = []
        ch = grid[r][c]
        if ch == "|" and (dr, dc) in (E, W):
            res.append(N)
            res.append(S)
        elif ch == "-" and (dr, dc) in (N, S):
            res.append(E)
            res.append(W)
        elif ch == "/":
            res.append((-dc, -dr))
        elif ch == "\\":
            res.append((dc, dr))
        else:
            res.append((dr, dc))
        return res
    def bfs(r, c, dr, dc):
        q = deque([(r, c, dr, dc)])
        vis = set()
        while q:
            r, c, dr, dc = q.popleft()
            if (r, c, dr, dc) in vis: continue
            vis.add((r, c, dr, dc))
            for ndr, ndc in get(r, c, dr, dc):
                nr, nc = r + ndr, c + ndc
                if in_bounds(nr, nc):
                    q.append((nr, nc, ndr, ndc))
        return len(set([(r, c) for r, c, _, _ in vis]))
    p1 = p2 = 0
    for r, c in product(range(n), repeat = 2):
        if r in rows:
            dr, dc = rows[r]
            p2 = max(p2, bfs(r, c, dr, dc))
        if c in columns:
            dr, dc = columns[c]
            if (r, c) == (0, 0): p1 = bfs(r, c, dr, dc)
            p2 = max(p2, bfs(r, c, dr, dc))
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 17

### Solution 1:  Dijkstra, memoization

```py
def dijkstra(grid):
    n = len(grid)
    in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
    neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    min_heap = [(0, 0, 0, 0, 0, 3)]
    memo = set()
    while min_heap:
        cost, r, c, dr, dc, steps = heapq.heappop(min_heap)
        if (r, c) == (n - 1, n - 1): return cost
        if (r, c, dr, dc, steps) in memo: continue
        memo.add((r, c, dr, dc, steps))
        if steps < 3:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                heapq.heappush(min_heap, (cost + grid[nr][nc], nr, nc, dr, dc, steps + 1))
        for ndr, ndc in neighborhood:
            if (ndr, ndc) == (dr, dc) or (ndr, ndc) == (-dr, -dc): continue # don't go forward or backward
            nr, nc = r + ndr, c + ndc
            if in_bounds(nr, nc):
                heapq.heappush(min_heap, (cost + grid[nr][nc], nr, nc, ndr, ndc, 1))
    return -1

with open("big.txt", "r") as f:
    grid = [list(map(int, row)) for row in f.read().splitlines()]
    assert len(grid) == len(grid[0])
    p1 = dijkstra(grid)
    p2 = 0
    print("part 1:", p1)
    print("part 2:", p2)
```

```py
def dijkstra(grid):
    n = len(grid)
    in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
    neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    min_heap = [(0, 0, 0, 0, 0, 10)]
    memo = set()
    while min_heap:
        cost, r, c, dr, dc, steps = heapq.heappop(min_heap)
        if (r, c) == (n - 1, n - 1): return cost
        if (r, c, dr, dc, steps) in memo: continue
        memo.add((r, c, dr, dc, steps))
        if steps < 10:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                heapq.heappush(min_heap, (cost + grid[nr][nc], nr, nc, dr, dc, steps + 1))
        if steps >= 4:
            for ndr, ndc in neighborhood:
                if (ndr, ndc) == (dr, dc) or (ndr, ndc) == (-dr, -dc): continue # don't go forward or backward
                nr, nc = r + ndr, c + ndc
                if in_bounds(nr, nc):
                    heapq.heappush(min_heap, (cost + grid[nr][nc], nr, nc, ndr, ndc, 1))
    return -1

with open("big.txt", "r") as f:
    grid = [list(map(int, row)) for row in f.read().splitlines()]
    assert len(grid) == len(grid[0])
    p2 = dijkstra(grid)
    p1 = 0
    print("part 1:", p1)
    print("part 2:", p2)
```

### Solution 2:  A* Search algorithm, heuristic, manhattan distance

```py

# f = g + h, cost + heuristic
def A_star_search(grid):
    n = len(grid)
    in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
    neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    min_heap = [(2 * n, 0, 0, 0, 0, 0, 10)]
    memo = set()
    def g(cost, r, c):
        return cost + grid[r][c]
    def h(r, c):
        return 2 * n - r - c
    def f(cost, r, c):
        return g(cost, r, c) + h(r, c)
    while min_heap:
        est_cost, cost, r, c, dr, dc, steps = heapq.heappop(min_heap)
        if (r, c) == (n - 1, n - 1): return cost
        if (r, c, dr, dc, steps) in memo: continue
        memo.add((r, c, dr, dc, steps))
        if steps < 10:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                heapq.heappush(min_heap, (f(cost, nr, nc), g(cost, nr, nc), nr, nc, dr, dc, steps + 1))
        if steps >= 4:
            for ndr, ndc in neighborhood:
                if (ndr, ndc) == (dr, dc) or (ndr, ndc) == (-dr, -dc): continue # don't go forward or backward
                nr, nc = r + ndr, c + ndc
                if in_bounds(nr, nc):
                    heapq.heappush(min_heap, (f(cost, nr, nc), g(cost, nr, nc), nr, nc, ndr, ndc, 1))
    return -1

with open("big.txt", "r") as f:
    grid = [list(map(int, row)) for row in f.read().splitlines()]
    assert len(grid) == len(grid[0])
    p2 = A_star_search(grid)
    p1 = 0
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 18

### Solution 1:  Picks' theorem, shoelace formula

```py
def shoelace(vertices):
    double_area = 0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        double_area += x1 * y2 - x2 * y1
    double_area = abs(double_area)
    return double_area // 2
directions = {
    "R": (0, 1),
    "L": (0, -1),
    "U": (-1, 0),
    "D": (1, 0)
}
edge = parse.compile("{} {:d} ({})")
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    r = c = b = 0
    vertices = [(r, c)]
    for line in data:
        dir, dist, hx = edge.parse(line).fixed
        dist = int(dist)
        b += dist
        dr, dc = directions[dir]
        r += dr * dist
        c += dc * dist
        vertices.append((r, c))
    vertices.pop()
    A = shoelace(vertices)
    p1 = A + b // 2 + 1
    print("part 1:", p1)
```

### Solution 2:  Pick's theorem, Shapely Polygon area

```py
directions = {
    "R": (0, 1),
    "L": (0, -1),
    "U": (-1, 0),
    "D": (1, 0)
}
edge = parse.compile("{} {:d} ({})")
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    r = c = b = 0
    vertices = [(r, c)]
    for line in data:
        dir, dist, hx = edge.parse(line).fixed
        dist = int(dist)
        b += dist
        dr, dc = directions[dir]
        r += dr * dist
        c += dc * dist
        vertices.append((r, c))
    vertices.pop()
    polygon = Polygon(vertices)
    A = int(polygon.area)
    p1 = A + b // 2 + 1
    print("part 1:", p1)
```

### Solution 3:  Part 2, convert hexadecimal to int

```py
directions = {
    "R": (0, 1),
    "L": (0, -1),
    "U": (-1, 0),
    "D": (1, 0)
}
dig_direction = "RDLU"
edge = parse.compile("{} {:d} ({})")
def plan(key):
    dist = int(key[1:-1], 16)
    dir = dig_direction[int(hx[-1])]
    return dir, dist
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    r = c = b = 0
    vertices = [(r, c)]
    for line in data:
        _, _, hx = edge.parse(line).fixed
        dir, dist = plan(hx)
        b += dist
        dr, dc = directions[dir]
        r += dr * dist
        c += dc * dist
        vertices.append((r, c))
    vertices.pop()
    polygon = Polygon(vertices)
    A = int(polygon.area)
    p2 = A + b // 2 + 1
    print("part 2:", p2)
```

## Day 19

### Solution 1:  directed graph, range splitting, regex, conditionals, control structure, dfs, recursion

```py
operators = {
    "<": operator.lt,
    ">": operator.gt
}
class Part:
    def __init__(self, x, m, a, s):
        self.ratings = {
            "x": x,
            "m": m,
            "a": a,
            "s": s
        }
    def sum(self):
        return sum(self.ratings.values())
class Workflow:
    def __init__(self, rules):
        rules = rules.split(",")
        self.default = rules.pop()
        self.conditions, self.responses = [], []
        for rule in rules:
            cond, resp = rule.split(":")
            self.conditions.append(cond)
            self.responses.append(resp)
    def evaluate(self, part):
        for cond, resp in zip(self.conditions, self.responses):
            op = re.search("[<>]", cond).group()
            category, value = cond.split(op)
            if operators[op](part.ratings[category], int(value)): return resp
        return self.default
    def __repr__(self):
        return f"Workflow({self.conditions}, {self.responses}, default = {self.default})"
def search(part, workflows):
    node = "in"
    while node not in "AR":
        node = workflows[node].evaluate(part)
    return node == "A"
categories = "xmas"
def dfs(name, ranges, workflows):
    if name in "AR":
        return math.prod(max(0, e - s) for s, e in ranges) if name == "A" else 0
    ans = 0
    workflow = workflows[name]
    for cond, resp in zip(workflow.conditions, workflow.responses):
        op = re.search("[<>]", cond).group()
        category, value = cond.split(op)
        value = int(value)
        category_index = categories.index(category)
        s, e = ranges[category_index]
        new_ranges = list(ranges)
        # x < 5, end point can be 4 at most, which is 5 in exclusive endpoint
        # x > 5, then you can start at 6
        if op == "<":
            left_seg = (s, min(e, value))
            right_seg = (max(s, value), e)
            new_ranges[category_index] = left_seg
            ranges[category_index] = right_seg
        else:
            left_seg = (s, min(e, value + 1))
            right_seg = (max(s, value + 1), e)
            ranges[category_index] = left_seg
            new_ranges[category_index] = right_seg
        ans += dfs(resp, new_ranges, workflows)
    return ans + dfs(workflow.default, ranges, workflows)
part_pat = parse.compile("{{x={:d},m={:d},a={:d},s={:d}}}")
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    pivot = data.index("")
    p1 = p2 = 0
    workflows = {}
    for line in data[:pivot]:
        name, rules = line.split("{")
        workflows[name] = Workflow(rules[:-1])
    for line in data[pivot + 1:]:
        x, m, a, s = part_pat.parse(line).fixed
        part = Part(x, m, a, s)
        if search(part, workflows):
            p1 += part.sum()
    ranges = [(1, 4001) for _ in range(4)]
    p2 = dfs("in", ranges, workflows)
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 20 

### Solution 1:  

```py
class ModuleType(StrEnum):
    BROADCASTER = "broadcaster"
    FLIPFLOP = "%"
    CONJUNCTION = "&"
class ModuleInterface(ABC):
    @abstractmethod
    def send(self, pulse):
        pass
    @abstractmethod
    def __repr__(self):
        pass
class BroadCaster(ModuleInterface):
    def send(self, pulse):
        return "low" 
    def __repr__(self):
        return "broadcaster"
class FlipFlop(ModuleInterface):
    def __init__(self):
        self.type = ModuleType.FLIPFLOP
        self.state = "off"
    def send(self, pulse):
        if pulse == "high": return None
        self.state = "on" if self.state == "off" else "off"
        return "low" if self.state == "off" else "high"
    def __repr__(self):
        return f"FlipFlop({self.state})"
class Conjunction(ModuleInterface):
    def __init__(self):
        self.type = ModuleType.CONJUNCTION
        self.state = {}
    def send(self, pulse):
        return "low" if all(prev == "high" for prev in self.state.values()) else "high"
    def __repr__(self):
        return f"Conjunction({self.state})"
class Module:
    def __new__(cls, type):
        if type == ModuleType.FLIPFLOP: return FlipFlop()
        elif type == ModuleType.CONJUNCTION: return Conjunction()
        elif type == ModuleType.BROADCASTER: return BroadCaster()
        else: raise Exception(f"unknown module type, {type}")
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    p1 = p2 = 0
    adj = {}
    modules = {}
    for line in data:
        if "broadcaster" in line:
            _, dest = line.split(" -> ")
            adj["broadcaster"] = dest.split(", ")
            modules["broadcaster"] = Module("broadcaster")
        else:
            src, dest = line.split(" -> ")
            adj[src[1:]] = dest.split(", ")
            modules[src[1:]] = Module(src[0])
    for u, dests in adj.items():
        for v in dests:
            if v in modules and modules[v].type == ModuleType.CONJUNCTION:
                modules[v].state[u] = "low"
    low_count = high_count = 0
    for _ in range(1_000):
        q = deque()
        # hit button 
        q.append(("broadcaster", "low"))
        while q:
            u, pulse = q.popleft()
            low_count += pulse == "low"
            high_count += pulse == "high"
            if u not in modules: continue
            pulse_out = modules[u].send(pulse)
            if pulse_out is None: continue
            for v in adj[u]:
                if v in modules and modules[v].type == ModuleType.CONJUNCTION:
                    modules[v].state[u] = pulse_out
                q.append((v, pulse_out))
    p1 = low_count * high_count
    print("part 1:", p1)
    print("part 2:", p2)
```

### Solution 1: 

```py
flip_flop = parse.compile("%{} -> {}")
conjunction = parse.compile("&{} -> {}")
broad = parse.compile("broadcaster -> {}")
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    start = "broadcaster"
    flop = "flip_flop"
    con = "conjunction"
    adj = defaultdict(list)
    module_type = {}
    seen = set()
    inputs = defaultdict(list)
    for line in data:
        if broad.parse(line):
            modules = broad.parse(line).fixed[0]
            for module in modules.split(", "):
                adj[start].append(module)
        elif flip_flop.parse(line):
            name, modules = flip_flop.parse(line).fixed
            module_type[name] = flop
            for module in modules.split(", "):
                adj[name].append(module)
        else:
            name, modules = conjunction.parse(line).fixed
            module_type[name] = con
            for module in modules.split(", "):
                adj[name].append(module)
    for u in module_type.keys():
        for v in adj[u]:
            if module_type.get(v, None) == con:
                inputs[v].append(u)
    last_pulse = {name: "low" for name in module_type.keys()}
    clicks = 10_000
    lcnt = hcnt = 0
    debug = [[] for _ in range(4)]
    state = {name: 0 for name, _ in module_type.items()}
    for i in range(clicks):
        if i == 1_000:  print("part 1:", lcnt * hcnt)
        in_queue = set([start])
        lcnt += 1
        queue = deque([(start, "low")])
        while queue:
            u, pulse = queue.popleft()
            if pulse == "low":
                lcnt += len(adj[u])
            else:
                hcnt += len(adj[u])
            for v in adj[u]:
                if v == "rx" and pulse == "low": print("part 2:", i)
                if module_type.get(v, None) == flop and pulse == "low":
                    state[v] ^= 1
                    if state[v] == 1: 
                        queue.append((v, "high"))
                        last_pulse[v] = "high"
                    else:
                        queue.append((v, "low"))
                        last_pulse[v] = "low"
                if module_type.get(v, None) == con:
                    if v == "ql":
                        for idx, node in enumerate(inputs[v]):
                            if last_pulse[node] == "high":
                                first[idx] = min(first[idx], i)
                        if all(v != math.inf for v in first): return 
                    if any(last_pulse[node] == "low" for node in inputs[v]):
                        queue.append((v, "high"))
                        last_pulse[v] = "high"
                    else:
                        queue.append((v, "low"))
                        last_pulse[v] = "low"
```

## Day 21: 

### Solution 1: 

```py
with open("big.txt", "r") as f:
    grid = [list(line) for line in f.read().splitlines()]
    R, C = len(grid), len(grid[0])
    neigborhood = lambda r, c: [(r + 1, c), (r, c + 1), (r - 1, c), (r, c - 1)]
    in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
    sr = sc = None
    for r, c in product(range(R), range(C)):
        if grid[r][c] == "S":
            sr, sc = r, c
    dist = [[math.inf] * C for _ in range(R)]
    queue = deque([(sr, sc, 0)])
    steps = 64
    # steps = 26501365
    while queue:
        r, c, d = queue.popleft()
        if dist[r][c] <= d: continue
        dist[r][c] = d
        for nr, nc in neigborhood(r, c):
            if not in_bounds(nr, nc) or grid[nr][nc] == "#": continue
            queue.append((nr, nc, d + 1))
    p1 = 0
    for r, c in product(range(R), range(C)):
        if dist[r][c] <= steps and dist[r][c] % 2 == steps % 2: p1 += 1
    print("part 1:", p1)
```

### Solution 2:  Modular arithmetic

```py
def solve(steps):
    with open("big.txt", "r") as f:
        grid = [list(line) for line in f.read().splitlines()]
        R, C = len(grid), len(grid[0])
        # assert it is a square grid
        assert R == C
        N = R
        neigborhood = lambda r, c: [(r + 1, c), (r, c + 1), (r - 1, c), (r, c - 1)]
        in_bounds = lambda r, c: 0 <= r < N and 0 <= c < N
        def bfs(r, c, corner = 0):
            dist = [[math.inf] * N for _ in range(N)]
            queue = deque([(r, c, corner)])
            while queue:
                r, c, d = queue.popleft()
                if dist[r][c] <= d: continue
                dist[r][c] = d
                for nr, nc in neigborhood(r, c):
                    if not in_bounds(nr, nc) or grid[nr][nc] == "#": continue
                    queue.append((nr, nc, d + 1))
            return dist
        def get_counts(sr, sc):
            dist = seeds[(sr, sc)]
            counts = [0] * 2
            for r, c in product(range(N), repeat = 2):
                if dist[r][c] == math.inf: continue
                counts[dist[r][c] % 2] += 1
            return counts
        seeds = {}
        for r, c in product(range(N), repeat = 2):
            if r == N // 2 or c == N // 2: assert grid[r][c] != "#"
            if grid[r][c] == "#": continue
            if grid[r][c] == "S" or (r, c) in [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1), (N // 2, 0), (N // 2, N - 1), (0, N // 2), (N - 1, N // 2)]:
                seeds[(r, c)] = bfs(r, c, corner = (0 if grid[r][c] == "S" else 1))
        res = sum(1 for r, c in product(range(N), repeat = 2) if grid[r][c] != "#" and seeds[(N // 2, N // 2)][r][c] % 2 == steps % 2 and seeds[(N // 2, N // 2)][r][c] <= steps)
        # four corners
        cnt = 0
        for r, c in [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1)]:
            counts = get_counts(r, c)
            dist = seeds[(r, c)]
            max_dist = max(max(filter(lambda x: x != math.inf, row)) for row in dist)
            cur_steps = 2 * (N // 2) + 1
            # print("res before", res)
            blocks = 0
            while cur_steps + max_dist < steps:
                blocks += 1
                if cur_steps % 2 == 0: res += blocks * counts[steps % 2]
                else: res += blocks * counts[(steps ^ 1) % 2]
                cur_steps += N
            while steps > cur_steps:
                cnt += 1
                if cnt % 1000 == 0: print(cnt)
                blocks += 1
                for r, c in product(range(N), repeat = 2):
                    if (dist[r][c] + cur_steps) % 2 == steps % 2 and dist[r][c] + cur_steps <= steps:
                        res += blocks
                cur_steps += N
            # print("res after", res)
        # four edges
        for r, c in [(N // 2, 0), (N // 2, N - 1), (0, N // 2), (N - 1, N // 2)]:
            counts = get_counts(r, c)
            dist = seeds[(r, c)]
            max_dist = max(max(filter(lambda x: x != math.inf, row)) for row in dist)
            cur_steps = N // 2
            # print("res before", res)
            # print("max", max_dist, "cur", cur_steps, "steps", steps)
            while steps > cur_steps + max_dist:
                # print("counts", counts)
                if cur_steps % 2 == 0: res += counts[steps % 2]
                else: res += counts[(steps ^ 1) % 2]
                cur_steps += N
            while steps > cur_steps:
                cnt += 1
                if cnt % 1000 == 0: print(cnt)
                for r, c in product(range(N), repeat = 2):
                    if (dist[r][c] + cur_steps) % 2 == steps % 2 and dist[r][c] + cur_steps <= steps:
                        res += 1
                cur_steps += N
            # print("res after", res)
    return res
def brute(steps):
    with open("big.txt", "r") as f:
        grid = [list(line) for line in f.read().splitlines()]
        R, C = len(grid), len(grid[0])
        # assert it is a square grid
        assert R == C
        N = R
        neigborhood = lambda r, c: [(r + 1, c), (r, c + 1), (r - 1, c), (r, c - 1)]
        queue = deque([(N // 2, N // 2, 0, 0)])
        for _ in range(steps + 1):
            nqueue = deque()
            seen = set()
            while queue:
                r, c, rr, cc = queue.popleft()
                if grid[r][c] == "#": continue
                if (r, c, rr, cc) in seen: continue
                seen.add((r, c, rr, cc))
                for nr, nc in neigborhood(r, c):
                    if nr == R: nqueue.append((0, nc, rr + 1, cc))
                    elif nr == -1: nqueue.append((R - 1, nc, rr - 1, cc))
                    elif nc == C: nqueue.append((nr, 0, rr, cc + 1))
                    elif nc == -1: nqueue.append((nr, C - 1, rr, cc - 1)) 
                    else: nqueue.append((nr, nc, rr, cc))
            queue = nqueue
    return len(seen)
# print("part 1:", brute(300))
# print("part 2:", solve(26501365))
print("part 1:", solve(64))
print("part 2:", solve(26501365))
```

### Unit testing brute force and optimized solution

```py
class Day21Test(unittest.TestCase):
    # testing on this input
    """
    ...........
    ......##.#.
    .###..#..#.
    ..#.#...#..
    ....#.#....
    .....S.....
    .##......#.
    .......##..
    .##.#.####.
    .##...#.##.
    ...........
    """
    def test_solve(self):
        for i in range(1, 100):
            self.assertEqual(brute(i), solve(i), f"Solutions don't match for i = {i}")
Day21Test().test_solve()
```

## Day 22: 

### Solution 1:  2d intersection, sets, bfs, queue

```py
coords = parse.compile("{:d},{:d},{:d}~{:d},{:d},{:d}")
with open("big.txt", "r") as f:
    bricks = sorted([list(coords.parse(line).fixed) for line in f.read().splitlines()], key = lambda x: x[2])
    def overlaps_1d(s1, s2, e1, e2):
        return min(e1, e2) - max(s1, s2) >= 0
    def overlaps_2d(brick1, brick2):
        return overlaps_1d(brick1[0], brick2[0], brick1[3], brick2[3]) and overlaps_1d(brick1[1], brick2[1], brick1[4], brick2[4])
    n = len(bricks)
    for i in range(n):
        max_z = 0
        for j in range(i):
            if overlaps_2d(bricks[i], bricks[j]):
                max_z = max(max_z, bricks[j][2], bricks[j][5])
        bricks[i][5] = bricks[i][5] - bricks[i][2] + max_z + 1
        bricks[i][2] = max_z + 1
    k_supports_v = [set() for _ in range(n)]
    v_supports_k = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i):
            if overlaps_2d(bricks[i], bricks[j]) and bricks[i][2] == bricks[j][5] + 1:
                k_supports_v[j].add(i)
                v_supports_k[i].add(j)
    p1 = p2 = 0
    for i in range(n):
        if all(len(v_supports_k[j]) >= 2 for j in k_supports_v[i]): 
            p1 += 1
            continue
        queue = deque([i])
        fallen = set([i])
        while queue:
            u = queue.popleft()
            for v in k_supports_v[u]:
                if len(v_supports_k[v] - fallen) == 0 and v not in fallen:
                    fallen.add(v)
                    queue.append(v)
        p2 += len(fallen) - 1
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 23: 

### Solution 1:  compress the graph into about 30 vertex, then the np hard solution becomes feasible.  longest path is an np hard problem.

```py
with open("small.txt", "r") as f:
    grid = [list(line) for line in f.read().splitlines()]
    R, C = len(grid), len(grid[0])
    print(R, C)
    sr = sc = 0
    for c in range(C):
        if grid[0][c] == ".":
            sr, sc = 0, c
    queue = deque([(sr, sc, 0, 0, 0)])
    in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
    neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    p1 = 0
    while queue:
        r, c, steps, pr, pc = queue.popleft()
        if r == R - 1: p1 = max(p1, steps)
        for nr, nc in neighborhood(r, c):
            if grid[r][c] == ">" and nc <= c: continue
            if grid[r][c] == "v" and nr <= r: continue
            if not in_bounds(nr, nc) or (nr, nc) == (pr, pc) or grid[nr][nc] == "#": continue
            queue.append((nr, nc, steps + 1, r, c))
    print("part 1:", p1)
```

```py
with open("big.txt", "r") as f:
    grid = [list(line) for line in f.read().splitlines()]
    R, C = len(grid), len(grid[0])
    sr = sc = er = ec = 0
    for c in range(C):
        if grid[0][c] == ".":
            sr, sc = 0, c
        if grid[-1][c] == ".":
            er, ec = R - 1, c
    in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
    neighborhood = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    vertices = {}
    for r, c in product(range(R), range(C)):
        if grid[r][c] == "#": continue
        nei_sum = sum(1 for nr, nc in neighborhood(r, c) if in_bounds(nr, nc) and grid[nr][nc] != "#")
        if nei_sum != 2:
            vertices[(r, c)] = len(vertices)
    n = len(vertices)
    stack = [(sr, sc, sr, sc, 0, 0)]
    adj = [[] for _ in range(n)]
    vis = set()
    while stack:
        r, c, pr, pc, u, w = stack.pop()
        if (r, c) in vertices and w > 0:
            v = vertices[(r, c)]
            adj[u].append((v, w))
            adj[v].append((u, w))
            u = v
            w = 0
        if (r, c) in vis: continue
        vis.add((r, c))
        for nr, nc in neighborhood(r, c):
            if not in_bounds(nr, nc) or (nr, nc) == (pr, pc) or grid[nr][nc] == "#": continue
            stack.append((nr, nc, r, c, u, w + 1))
    vis = [0] * n
    vis[0] = 1
    p2 = 0
    cur_w = 0
    def dfs(u):
        global cur_w, p2
        if u == n - 1:
            p2 = max(p2, cur_w)
            return
        for v, w in adj[u]:
            if vis[v]: continue
            cur_w += w
            vis[v] = 1
            dfs(v)
            cur_w -= w
            vis[v] = 0
    dfs(0)
    print("part 2:", p2)
```

## Day 24: 

### Solution 1:  linear algebra, vectors, 2d, system of nonlinear equations, sympy solver

```py
class Line:
    def __init__(self, m, b, px, py, vx, vy):
        self.m = m
        self.b = b
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
    def eval(self, x):
        return self.m * x + self.b
    def __repr__(self):
        return f"m = {self.m}, b = {self.b}, px = {self.px}, py = {self.py}"
hailstone = parse.compile("{:d}, {:d}, {:d} @ {:d}, {:d}, {:d}")
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    small, big = 200000000000000, 400000000000000
    hail = []
    for line in data:
        px, py, pz, vx, vy, vz = hailstone.parse(line).fixed
        m = vy / vx
        b = py - m * px
        hail.append(Line(m, b, px, py, vx, vy))
    p1 = 0
    n = len(hail)
    in_bounds = lambda x, y: small <= x <= big and small <= y <= big
    def intersection(l1, l2):
        if l1.m == l2.m: return 0, 0
        x = (l2.b - l1.b) / (l1.m - l2.m)
        y = l1.eval(x)
        return x, y
    def in_past(x, y, l):
        return (l.vx > 0 and x < l.px) or (l.vx < 0 and x > l.px) or (l.vy > 0 and y < l.py) or (l.vy < 0 and y > l.py)
    for i in range(n):
        for j in range(i + 1, n):
            x, y = intersection(hail[i], hail[j])
            if in_past(x, y, hail[i]) or in_past(x, y, hail[j]): continue
            if in_bounds(x, y): p1 += 1
    print(p1)
```

```py

hailstone = parse.compile("{:d}, {:d}, {:d} @ {:d}, {:d}, {:d}")
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    # small, big = 200000000000000, 400000000000000
    x_r, y_r, z_r, vx_r, vy_r, vz_r = sympy.symbols("x_r, y_r, z_r, vx_r, vy_r, vz_r")
    equations = []
    for line in data[:6]:
        px, py, pz, vx, vy, vz = hailstone.parse(line).fixed
        equations.append((px - x_r) * (vy_r - vy) - (py - y_r) * (vx_r - vx))
        equations.append((pz - z_r) * (vy_r - vy)  - (py - y_r) * (vz_r - vz))
    solution = sympy.solve(equations, [x_r, y_r, z_r, vx_r, vy_r, vz_r], dict = True)[0]
    p2 = sum(solution[var] for var in [x_r, y_r, z_r])
    print("part 2:", p2)
```

## Day 25: 

max flow min cut algorithms

### Solution 1:  undirected graph, connected components, disjoint set union

```py
net = Network(notebook = True)
net.from_nx(G)
net.show_buttons(filter_=['physics'])
net.show("small.html")
```

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    nodes = set()
    edges = []
    for line in data:
        u, nei_nodes = line.split(": ")
        nodes.add(u)
        for v in nei_nodes.split():
            edges.append((u, v))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    G.remove_edges_from([("zjm", "zcp"), ("rfg", "jks"), ("nsk", "rsg")])
    p1 = math.prod(len(s) for s in nx.connected_components(G))
    print("part 1:", p1)
```

### Solution 2:  networkx minimum cut to partition into two disconnected graphs

minimum cut, until the cut value is 3, which means the maximum flow is equal to 3 with the current source and target node. 

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    nodes = set()
    edges = []
    for line in data:
        u, nei_nodes = line.split(": ")
        nodes.add(u)
        for v in nei_nodes.split():
            edges.append((u, v))
    nodes = list(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges, capacity = 1)
    u = nodes[0]
    for v in nodes[1:]:
        cut_value, partitions = nx.minimum_cut(G, u, v)
        if cut_value == 3: 
            print("part 1:", math.prod(len(p) for p in partitions))
            break
```

### Solution 3:  networkx minimum edge cut, 

it returns the edges to cut for minimum cut, so you just keep doing it until it is 3 edges that are cut.  In my test data set it happens in the first attempt

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    nodes = set()
    edges = []
    for line in data:
        u, nei_nodes = line.split(": ")
        nodes.add(u)
        for v in nei_nodes.split():
            edges.append((u, v))
    nodes = list(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges, capacity = 1)
    u = nodes[0]
    for v in nodes[1:]:
        cuts = nx.minimum_edge_cut(G, u, v)
        if len(cuts) == 3:
            G.remove_edges_from(cuts)
            print("part 1:", math.prod(map(len, nx.connected_components(G))))
            break
```

You can also do it without specifying the source and target node and it looks for the solution that disconnects the graph into two partitions with minimum cardinality for the edges.  So it finds the solution that cuts the fewest edges, since each edge represents capacity = 1.  This should be the three edges, so then just need to remove those edges.  Then it disconnected the graph and take the product of the connected components of the remaining graph.

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    nodes = set()
    edges = []
    for line in data:
        u, nei_nodes = line.split(": ")
        nodes.add(u)
        for v in nei_nodes.split():
            edges.append((u, v))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges, capacity = 1)
    cuts = nx.minimum_edge_cut(G)
    G.remove_edges_from(cuts)
    print("part 1:", math.prod(map(len, nx.connected_components(G))))
```

### Solution 4:  Stoer Wagner algorithm for global minimum cut

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    nodes = set()
    edges = []
    for line in data:
        u, nei_nodes = line.split(": ")
        nodes.add(u)
        for v in nei_nodes.split():
            edges.append((u, v))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges, weight = 1)
    cut_value, partition = nx.stoer_wagner(G)
    print("part 1:", math.prod(map(len, partition)))
```

### Solution 5:  From Scratch Stoer Wagner Algorithm for global minimum cut

```py
def stoer_wagner(nodes, adj, edges):
    n = len(nodes)
    min_cut = math.inf
    connected_components = {node: [node] for node in nodes}
    partition_size = None
    prev = 0
    index_map = {node: i for i, node in enumerate(nodes)}
    vis = [0] * n
    for i in range(n - 1):
        # not necessarily random, and doesn't have to be
        u = prev
        while vis[u]: u += 1
        prev = u
        u = nodes[u]
        seen = set()
        weights = Counter()
        max_heap = []
        while len(seen) < n - i - 1:
            while max_heap and weights[max_heap[0][1]] != abs(max_heap[0][0]):
                heapq.heappop(max_heap)
            if max_heap:
                u = heapq.heappop(max_heap)[1]
                weights[u] = 0
            seen.add(u)
            for v in adj[u]:
                if v in seen: continue
                w = edges[(u, v)]
                weights[v] += w
                heapq.heappush(max_heap, (-weights[v], v))
        while max_heap and weights[max_heap[0][1]] != abs(max_heap[0][0]):
            heapq.heappop(max_heap)
        assert len(max_heap) > 0, "max heap is empty"
        wei, v = heapq.heappop(max_heap)
        wei = abs(wei)
        if wei < min_cut:
            min_cut = wei
            partition_size = len(connected_components[v])
        # contract v into u, merge into one connected component
        connected_components[u].extend(connected_components[v])
        # loop through children of v as w
        neighbors = adj[v]
        for w in neighbors:
            adj[w].remove(v)
            if w == u: continue
            if (u, w) not in edges:
                adj[u].add(w)
                adj[w].add(u)
            edges[(u, w)] += edges[(v, w)]
            edges[(w, u)] += edges[(w, v)]
        adj[v].clear()
        vis[index_map[v]] = 1
    return min_cut, partition_size

with open("big.txt", "r") as f:
    data = f.read().splitlines()
    nodes = set()
    adj = defaultdict(set)
    edges = Counter()
    for line in data:
        u, nei_nodes = line.split(": ")
        nodes.add(u)
        for v in nei_nodes.split():
            adj[u].add(v)
            adj[v].add(u)
            edges[(u, v)] = 1
            edges[(v, u)] = 1
            nodes.add(v)
    n = len(nodes)
    nodes = list(nodes)
    cut_value, partition_size = stoer_wagner(nodes, adj, edges)
    other_partition_size = n - partition_size
    print("part 1:", partition_size * other_partition_size)
```

```py
def stoer_wagner(nodes, adj, edges):
    contractions = []
    n = len(nodes)
    min_cut = math.inf
    best_phase = 0
    original_nodes = copy.deepcopy(nodes)
    # G = nx.Graph()
    # for (u, v), w in edges.items():
    #     G.add_edge(u, v, weight = w, label = str(w), title = str(w))
    # net = Network(notebook = True)
    # net.from_nx(G)
    # net.show_buttons(filter_=['physics'])
    # net.show(f"graph/small_0000.html")
    for i in range(n - 1):
        # not necessarily random, and doesn't have to be
        u = nodes.pop()
        nodes.add(u)
        seen = set()
        values = Counter()
        max_heap = []
        while len(seen) < n - i - 1:
            while max_heap and values[max_heap[0][1]] != abs(max_heap[0][0]):
                heapq.heappop(max_heap)
            if max_heap:
                u = heapq.heappop(max_heap)[1]
                values[u] = 0
            seen.add(u)
            for v in adj[u]:
                if v in seen or v not in nodes: continue
                w = edges[(u, v)]
                values[v] += w
                heapq.heappush(max_heap, (-values[v], v))
        while max_heap and values[max_heap[0][1]] != abs(max_heap[0][0]):
            heapq.heappop(max_heap)
        assert len(max_heap) > 0, "max heap is empty"
        wei, v = heapq.heappop(max_heap)
        wei = abs(wei)
        # print(values)
        # print("wei", wei, "v", v, "counter v", values[v])

        if wei < min_cut:
            min_cut = wei
            best_phase = i
        # contract v into u
        contractions.append((u, v))
        # loop through children of v as w
        for w in adj[v]:
            if w == u: continue
            if (u, w) not in edges:
                adj[u].append(w)
                adj[w].append(u)
            edges[(u, w)] += edges[(v, w)]
            edges[(w, u)] += edges[(w, v)]
            # print("u", u, "w", w, "edges", edges[(u, w)], edges[(w, u)])
        nodes.remove(v)
        # G = nx.Graph()
        # for u in nodes:
        #     for v in adj[u]:
        #         if v not in nodes: continue
        #         G.add_edge(u, v, weight = edges[(u, v)], label = str(edges[(u, v)]), title = str(edges[(u, v)]))
        # net = Network(notebook = True)
        # net.from_nx(G)
        # net.show_buttons(filter_=['physics'])
        # net.show(f"graph/small_{str(i + 1).zfill(4)}.html")
    p1, p2 = set(), set()
    for u, v in contractions[:best_phase]:
        p1.update((u, v))
    p2 = original_nodes - p1
    print(best_phase)
    return min_cut, [p1, p2]


with open("small.txt", "r") as f:
    data = f.read().splitlines()
    nodes = set()
    adj = defaultdict(list)
    edges = Counter()
    for line in data:
        u, nei_nodes = line.split(": ")
        nodes.add(u)
        for v in nei_nodes.split():
            adj[u].append(v)
            adj[v].append(u)
            edges[(u, v)] = 1
            edges[(v, u)] = 1
            nodes.add(v)
    cut_value, partition = stoer_wagner(nodes, adj, edges)
    print(cut_value, partition)
    # G = nx.Graph()
    # G.add_nodes_from(nodes)
    # G.add_edges_from(edges, weight = 1)
    # cut_value, partition = nx.stoer_wagner(G)
    print("part 1:", math.prod(map(len, partition)))
```

### Solution 6:  max flow min cut theorem, max flow with dinics algorithm, reconstruct partition by visiting all nodes from unsaturated edges.

```py
class FordFulkersonMaxFlow:
    """
    Ford-Fulkerson algorithm 
    - pluggable augmenting path finding algorithms
    - residual graph
    - bottleneck capacity
    """
    def __init__(self, n: int, edges: List[Tuple[int, int, int]]):
        self.size = n
        self.edges = edges
        self.cap = defaultdict(Counter)
        self.flow = defaultdict(Counter)
        self.adj_list = [[] for _ in range(self.size)]

    def build(self) -> None:
        self.delta = 0
        for src, dst, cap in self.edges:
            self.cap[src][dst] += cap
            self.adj_list[src].append(dst)
            self.adj_list[dst].append(src) # residual edge
            self.delta = max(self.delta, self.cap[src][dst])
        highest_bit_set = self.delta.bit_length() - 1
        self.delta = 1 << highest_bit_set

    def residual_capacity(self, src: int, dst: int) -> int:
        return self.cap[src][dst] - self.flow[src][dst]

    def main_dfs(self, source: int, sink: int) -> int:
        self.build()
        maxflow = 0
        while True:
            self.reset()
            cur_flow = self.dfs(source, sink, math.inf)
            if cur_flow == 0:
                break 
            maxflow += cur_flow
        return maxflow

    def neighborhood(self, node: int) -> List[int]:
        return (i for i in self.adj_list[node])

    def dinics_bfs(self, source: int, sink: int) -> bool:
        self.distances = [-1] * self.size
        self.distances[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for nei in self.neighborhood(node):
                if self.distances[nei] == -1 and self.residual_capacity(node, nei) > 0:
                    self.distances[nei] = self.distances[node] + 1
                    queue.append(nei)
        return self.distances[sink] != -1

    def dinics_dfs(self, node: int, sink: int, flow: int) -> int:
        if flow == 0: return 0
        if node == sink: return flow
        while self.ptr[node] < len(self.adj_list[node]):
            nei = self.adj_list[node][self.ptr[node]]
            self.ptr[node] += 1
            if self.distances[nei] == self.distances[node] + 1 and self.residual_capacity(node, nei) > 0:
                cur_flow = self.dinics_dfs(nei, sink, min(flow, self.residual_capacity(node, nei)))
                if cur_flow > 0:
                    self.flow[node][nei] += cur_flow
                    self.flow[nei][node] -= cur_flow
                    return cur_flow
        return 0

    def main_dinics(self, source: int, sink: int) -> int:
        self.build()
        maxflow = 0
        while self.dinics_bfs(source, sink):
            self.ptr = [0] * self.size # pointer to the next edge to be processed (optimizes for dead ends)
            while True:
                cur_flow = self.dinics_dfs(source, sink, math.inf)
                if cur_flow == 0:
                    break
                maxflow += cur_flow
        return maxflow
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    nodes = []
    index_map = {}
    edges = []
    for line in data:
        u, nei_nodes = line.split(": ")
        if u not in index_map:
            index_map[u] = len(index_map)
            nodes.append(u)
        for v in nei_nodes.split():
            if v not in index_map:
                index_map[v] = len(index_map)
                nodes.append(v)
            edges.append((index_map[u], index_map[v], 1))
            edges.append((index_map[v], index_map[u], 1))
    n = len(nodes)
    source = 0
    for sink in range(1, n):
        maxflow = FordFulkersonMaxFlow(n, edges)
        mf = maxflow.main_dinics(source, sink)
        if mf == 3:
            partition = set()
            queue = deque([source])
            while queue:
                u = queue.popleft()
                if u in partition: continue
                partition.add(u)
                for v in maxflow.adj_list[u]:
                    if maxflow.flow[u][v] < maxflow.cap[u][v]: queue.append(v)
            p1 = len(partition)
            p2 = n - p1
            print("part 1:", p1 * p2)
            break
```

### Solution 7:  Edmond's Karp Algorithm

```py

edmon```