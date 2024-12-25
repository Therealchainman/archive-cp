# Advent of Code 2024

## Day 1

### Solution 1:  sorting, counter

```py
A, B = [], []
with open("big.txt", "r") as f:
    for line in f:
        left, right = map(int, line.split()) 
        A.append(left)
        B.append(right)
    A.sort()
    B.sort()
    p1 = 0
    for x, y in zip(A, B):
        p1 += abs(x - y)
    p2 = 0
    a = Counter(A)
    b = Counter(B)
    for v in A:
        p2 += v * b[v]
    print("part 1:", p1)
    print("part 2:", p2)
```

## Day 2

### Solution 1:  strictly increasing array

1. The trick is to just try the original data and reversed, so only need the logic for strictly increasing array.
1. Then encode that and include the logic to allow for one violation, and check if that is removed if it become strictly increasing

```py
from typing import List

def is_valid(row: List[int]) -> bool:
    """
    Check if a row satisfies gradient conditions in original or reversed order.
    """
    return is_gradient(copy.deepcopy(row)) or is_gradient(row[::-1])

def find_marked_indices(data: List[int], difference_max: int = 3) -> List[int]:
    """
    Find indices where the difference between consecutive elements exceeds the threshold.
    """
    return [i for i in range(1, len(data)) if not (1 <= data[i] - data[i - 1] <= difference_max)]

def is_increasing_with_limit(data: List[int], limit: int) -> bool:
    """
    Check if the data is strictly increasing, with differences within the specified limit.
    """
    return all(data[i] > data[i - 1] and data[i] - data[i - 1] <= limit for i in range(1, len(data)))

def remove_and_check(data: List[int], index: int, limit: int) -> bool:
    """
    Remove an element at the specified index, check if the remaining list satisfies conditions,
    and reinsert the element.
    """
    removed_value = data.pop(index)
    is_valid = is_increasing_with_limit(data, limit)
    data.insert(index, removed_value)
    return is_valid

def is_gradient(data: List[int], difference_threshold: int = 3) -> bool:
    """
    Check if the data satisfies gradient conditions, allowing one violation.
    """
    marked_indices = find_marked_indices(data, difference_threshold)
    if not marked_indices: return True
    bad_index = marked_indices[0]
    for i in range(bad_index - 1, bad_index + 1):
        if i < 0: continue
        if remove_and_check(data, i, difference_threshold): return True
    return False

def calculate(filename: str):
    """
    Read a grid from a file and count rows that satisfy the gradient conditions.
    """
    with open(filename, "r") as f:
        grid = [list(map(int, line.split())) for line in f.read().splitlines()]
    ans = sum(1 for row in grid if is_valid(row))
    print(ans)
calculate("small.txt")
calculate("big.txt")
```

## Day 3

### Solution 1:  regex, re finditer, capture groups

1. The things to practice is that re.finditer will find all matches to the pattern.
1. The pattern is designed to match multiple patterns by using the logical or operator | to separate subpatterns so if it matches any it will return a match
1. Then they will be matched in order in the re.finditer, so then just need to check if the match substring with group() is do or don't
1. And then use the capturing groups to get the numbers.

```py
import re
def calculate(filename: str):
    """
    Read a grid from a file and count rows that satisfy the gradient conditions.
    """
    with open(filename, "r") as f:
        data = f.read()
        enabled = 1
        ans = 0
        pattern = r"mul\((\d{1,3}),(\d{1,3})\)|do\(\)|don't\(\)"
        for match in re.finditer(pattern, data):
            if match.group() == "do()":
                enabled |= 1
            elif match.group() == "don't()":
                enabled &= 0
            else:
                if enabled:
                    x, y = map(int, match.groups())
                    ans += x * y
    print(ans)
calculate("small.txt")
calculate("big.txt")
```

## Day 4

### Solution 1:  grid, diagonals in grid

```py
from enum import Enum
from collections import Counter
from typing import List, Generator, Tuple
from itertools import product

class Xmas(Enum):
    """
    Enum representing different symbols in the grid and their corresponding integer values.
    """
    M = 0  # Represents the "M" symbol.
    S = 1  # Represents the "S" symbol.
    A = 2  # Represents the "A" symbol.
    X = 3  # Represents the "X" symbol.

class XmasGrid:
    """
    A class to handle loading, processing, and analyzing a grid for specific patterns.
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty grid with dimensions set to zero.
        """
        self.R: int = 0  # Number of rows in the grid
        self.C: int = 0  # Number of columns in the grid
        self.grid: List[str] = []  # The grid itself, stored as a list of strings
    
    def load(self, filename: str) -> None:
        """
        Load the grid from a file.

        Args:
            filename (str): The name of the file containing the grid.
        """
        with open(filename, "r") as f:
            self.grid = f.read().splitlines()
            self.R = len(self.grid)
            self.C = len(self.grid[0]) if self.R > 0 else 0
    
    def in_bounds(self, r: int, c: int) -> bool:
        """
        Check if a given coordinate is within the bounds of the grid.

        Args:
            r (int): Row index.
            c (int): Column index.

        Returns:
            bool: True if the coordinate is within bounds, False otherwise.
        """
        return 0 <= r < self.R and 0 <= c < self.C
    
    def diagonal_vectors(self, r: int, c: int) -> Generator[Tuple[int, int], None, None]:
        """
        Generate all diagonal vectors (dr, dc) for a given cell.

        Args:
            r (int): Row index.
            c (int): Column index.

        Yields:
            Tuple[int, int]: The change in row and column for diagonal neighbors.
        """
        for dr, dc in product(range(-1, 2), repeat=2):
            if abs(dr) + abs(dc) < 2:  # Skip non-diagonal and the origin
                continue
            yield dr, dc
    
    def diagonal_match(self, r: int, c: int) -> bool:
        """
        Check if the cell at (r, c) satisfies the diagonal match condition.

        Args:
            r (int): Row index.
            c (int): Column index.

        Returns:
            bool: True if the diagonal match condition is met, False otherwise.
        """
        if self.grid[r][c] != "A":
            return False
        
        diagonal_counters: Counter[int] = Counter()
        
        for dr, dc in self.diagonal_vectors(r, c):
            nr, nc = r + dr, c + dc
            if not self.in_bounds(nr, nc):
                return False
            symbol = self.grid[nr][nc]
            if symbol not in Xmas.__members__:
                return False
            diagonal_counters[abs(dr + dc)] += Xmas[symbol].value
        
        # Ensure all diagonal counters have a value of 1
        return all(v == 1 for v in diagonal_counters.values())
    
    def calculate(self, filename: str) -> None:
        """
        Process the grid and count the number of cells that satisfy the diagonal match condition.

        Args:
            filename (str): The name of the file containing the grid.
        """
        ans: int = 0
        self.load(filename)
        for r, c in product(range(self.R), range(self.C)):
            ans += self.diagonal_match(r, c)
        print(ans)

# Example Usage
xmas_grid = XmasGrid()
xmas_grid.calculate("big.txt")
```

## Day 5

### Solution 1:  directed graph, topological ordering, queue, induced subgraph

TODO

```py
from collections import Counter
from parse import compile

def is_ordered(update, rules):
    position = {page: idx for idx, page in enumerate(update)}
    for X, Y in rules:
        if X in position and Y in position:
            if position[X] >= position[Y]:
                return False
    return True

def answer(suc, update, rules):
    if is_ordered(update, rules): return 0
    ind = Counter()
    for i in range(len(update)):
        for j in range(i + 1, len(update)):
            u, v = update[i], update[j]
            if v in suc[u]: ind[v] += 1
            if u in suc[v]: ind[u] += 1
    ans = []
    q = deque([x for x in update if ind[x] == 0])
    while q:
        u = q.popleft()
        ans.append(u)
        for v in suc[u]:
            if ind[v] > 0:
                ind[v] -= 1
                if ind[v] == 0:
                    q.append(v)
    return ans[len(ans) // 2]

def calculate(filename: str):
    with open(filename, "r") as f:
        data = f.read().splitlines()
        ans = 0
        pat = compile("{:d}|{:d}")
        rules = []
        suc = defaultdict(list)
        for line in data:
            if pat.parse(line) is not None:
                x, y = pat.parse(line)
                rules.append((x, y))
                suc[x].append(y)
            elif line:
                arr = list(map(int, line.split(",")))
                ans += answer(suc, arr, rules)
    print(ans)
calculate("small.txt")
calculate("big.txt")
```

## Day 6

### Solution 1:  grid traversal, set, grid

TODO

```py
from itertools import product

class PatrolGrid:
    DIR = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    R = C = 0
    grid = []

    def in_bounds(self, r, c):
        return 0 <= r < self.R and 0 <= c < self.C
    
    def is_loop(self, r, c):
        vis = set()
        p = 0
        while self.in_bounds(r, c):
            if (r, c, p) in vis: return True
            vis.add((r, c, p))
            dr, dc = self.DIR[p]
            r += dr
            c += dc
            if self.in_bounds(r, c) and self.grid[r][c] == "#":
                r -= dr
                c -= dc
                p = (p + 1) % len(self.DIR)
        return False

    def calculate(self, filename):
        with open(filename, "r") as f:
            self.grid = [list(x) for x in f.read().splitlines()]
            self.R, self.C = len(self.grid), len(self.grid[0])
            vis = set()
            sr = sc = 0
            for r, c in product(range(self.R), range(self.C)):
                if self.grid[r][c] == "^":
                    sr, sc = r, c
                    break
            p = ans = 0
            while self.in_bounds(r, c):
                vis.add((r, c))
                dr, dc = self.DIR[p]
                r += dr
                c += dc
                if self.in_bounds(r, c) and self.grid[r][c] == "#":
                    r -= dr
                    c -= dc
                    p = (p + 1) % len(self.DIR)
            for r, c in vis:
                self.grid[r][c] = "#"
                ans += self.is_loop(sr, sc)
                self.grid[r][c] = "."
            print(ans)
PatrolGrid().calculate("big.txt")
```

## Day 7

### Solution 1:  recursion, arithmetic evaluation, brute force

```py
def calculate(filename):
    def recurse(i, val):
        if i == len(nums): 
            return val == target
        if recurse(i + 1, val + nums[i]): return True
        if recurse(i + 1, val * nums[i]): return True
        if recurse(i + 1, int(str(val) + str(nums[i]))): return True
        return False
    with open(filename, "r") as f:
        data = f.read().splitlines()
        ans = 0
        for line in data:
            arr = line.split()
            target = int(arr[0][:-1])
            arr.pop(0)
            nums = list(map(int, arr))
            if recurse(1, nums[0]): ans += target
        print(ans)
calculate('big.txt')
```

## Day 8

### Solution 1:  grid, visited, vectors, lines

```py
def calculate(filename):
    def in_bounds(r, c):
        return 0 <= r < N and 0 <= c < N
    def fill_line(r, c, r1, c1):
        dr, dc = r1 - r, c1 - c
        while in_bounds(r, c):
            vis[r][c] = True
            r += dr
            c += dc
    with open(filename, "r") as f:
        grid = [list(line) for line in f.read().splitlines()]
        N = len(grid)
        vis = [[False for _ in range(N)] for _ in range(N)]
        ant = defaultdict(list)
        for r, c in product(range(N), repeat = 2):
            if grid[r][c] != ".":
                ant[grid[r][c]].append((r, c))
        for vec in ant.values():
            for i in range(len(vec)):
                for j in range(i):
                    r1, c1 = vec[i]
                    r2, c2 = vec[j]
                    fill_line(r1, c1, r2, c2)
                    fill_line(r2, c2, r1, c1)
        ans = sum(sum(row) for row in vis)
        print(ans)
calculate("small.txt")
calculate("big.txt")
```

## Day 9

### Solution 1: 

TODO

Need to do clean up on this solution 

```py
from sortedcontainers import SortedList
class SegmentTree:
    def __init__(self, n, neutral, func):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)] 

    def ascend(self, segment_idx):
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx, val):
        segment_idx += self.size - 1
        self.nodes[segment_idx] = val
        self.ascend(segment_idx)
            
    def query(self, left, right):
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.func(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self):
        return f"nodes array: {self.nodes}"
def calculate(filename):
    with open(filename, "r") as f:
        data = f.read()
        arr = list(map(int, data))
        N = sum(arr)
        pos = [-1] * N
        blocks_location = {}
        blocks = []
        updates = {}
        updated_blocks = set()
        size_map = {}
        free_space = [SortedList() for _ in range(N)]
        seg = SegmentTree(N, math.inf, min)
        j = 0
        for i in range(len(arr)):
            if i % 2 == 0:
                assert(arr[i] > 0)
                blocks_location[i // 2] = j
                for _ in range(arr[i]):
                    pos[j] = i // 2
                    j += 1
                blocks.append(arr[i])
            else:
                free_space[arr[i]].add(j)
                size_map[j] = arr[i]
                if len(free_space[arr[i]]) == 1: 
                    seg.update(arr[i], j)
                j += arr[i]
        # print(size_map)
        for i in range(len(blocks) - 1, 0, -1):
            p = seg.query(blocks[i], N + 1)
            # print(i, blocks[i], p)
            if p >= blocks_location[i]: continue
            assert(blocks_location[i] > p)
            free_space_size = size_map[p]
            free_space[free_space_size].pop(0)
            if len(free_space[free_space_size]) > 0:
                seg.update(free_space_size, free_space[free_space_size][0])
            else:
                seg.update(free_space_size, math.inf)
            rem_space = free_space_size - blocks[i]
            updates[p] = i
            p += blocks[i]
            if rem_space > 0:
                size_map[p] = rem_space
                free_space[rem_space].add(p)
                seg.update(rem_space, free_space[rem_space][0])
            updated_blocks.add(i)
        res = [-1] * N
        for k, v in updates.items():
            for i in range(blocks[v]):
                res[k + i] = v
        for i in range(N):
            if pos[i] == -1: continue
            if pos[i] not in updated_blocks:
                res[i] = pos[i]
        # j = N - 1
        # for i in range(N):
        #     if pos[i] == -1:
        #         while j > i and pos[j] == -1:
        #             j -= 1
        #         if j <= i: break
        #         pos[i], pos[j] = pos[j], pos[i]
        ans = 0
        for i in range(N):
            if res[i] == -1: continue
            ans += res[i] * i
        print(ans)

# calculate("small.txt")
calculate("big.txt")
```

## Day 10

### Solution 1:  grid, topographic map, multisource bfs, dynamic programming, memoization

```py
def calculate(filename):
    def in_bounds(r, c):
        return 0 <= r < N and 0 <= c < N
    def neighborhood(r, c):
        return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    with open(filename, "r") as f:
        grid = [list(map(int, row)) for row in f.read().splitlines()]
        N = len(grid)
        dp = [[0] * N for _ in range(N)]
        q = deque()
        for r, c in product(range(N), repeat = 2):
            if grid[r][c] == 0:
                dp[r][c] = 1
                q.append((r, c))
        while q:
            r, c = q.popleft()
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or grid[nr][nc] != grid[r][c] + 1: continue
                if dp[nr][nc] == 0: q.append((nr, nc))
                dp[nr][nc] += dp[r][c]
        ans = 0
        for r, c in product(range(N), repeat = 2):
            if grid[r][c] == 9: ans += dp[r][c]
        print(ans)
calculate("small.txt")
calculate("big.txt")
```

## Day 11

### Solution 1:  dynamic programming, functools.cache

```py
def calculate(filename):
    @cache
    def dp(i, val):
        if i == 75: return 1
        if val == 0:
            return dp(i + 1, 1)
        elif len(str(val)) % 2 == 0:
            x, y = split(val)
            return dp(i + 1, x) + dp(i + 1, y)
        return dp(i + 1, val * 2024)
    def split(x):
        s = str(x)
        n = len(s)
        return int(s[: n // 2]), int(s[n // 2:])
    with open(filename, "r") as f:
        data = list(map(int, f.read().split()))
        ans = sum(dp(0, val) for val in data)
        print(ans)
calculate("big.txt")
```

### Solution 2:  dynamic programming, counts, math, logarithm

1. Found this cool math trick that allows you to get the number of digits by taking the log base 10 of number
1. And the k and is to short-circuit the k = 0 case, cause in that scenario log is undefined.
1. And then there is a math way to get the first and second half based on number of digits and 10 to the power of half that. 

```py
def calculate(filename):
    with open(filename, "r") as f:
        data = list(map(int, f.read().split()))
        counts = Counter(data)
        for i in range(75):
            new_counts = Counter()
            for k, v in counts.items():
                n = k and int(math.log10(k) + 1)
                if k == 0:
                    new_counts[1] += v
                elif n % 2 == 0:
                    p = n // 2
                    x, y = k // 10 ** p, k % 10 ** p
                    new_counts[x] += v
                    new_counts[y] += v
                else:
                    new_counts[k * 2024] += v
            counts = new_counts
        ans = sum(v for v in counts.values())
        print(ans)
calculate("small.txt")
calculate("big.txt")
```

## Day 12

### Solution 1:  grid, flood fill, dfs, connected components in grid

1. I use a trick to count sides by counting the corners.  Detecting corners can be done with a kernel of size 4, so you just need to look at 4 elements. 
1. Adding the outside buffer makes the algorithm work without edge cases that would make it more complicated.

```py
def calculate(filename):
    def neighborhood(r, c):
        return [(r - 1, c), (r + 1, c), (r, c + 1), (r, c - 1)]
    def in_bounds(r, c):
        return 0 <= r < N and 0 <= c < N
    def dfs(r, c):
        if (r, c) in components: return
        components[(r, c)] = comp_id
        for nr, nc in neighborhood(r, c):
            if not in_bounds(nr, nc) or grid[r][c] != grid[nr][nc]: continue
            dfs(nr, nc)
    with open(filename, "r") as f:
        grid = [list("#" + row + "#") for row in f.read().splitlines()]
        N = len(grid[0])
        grid.insert(0, ["#"] * N)
        grid.append(["#"] * N)
        ans = comp_id = 0
        components = {}
        side_counts = Counter()
        for r, c in product(range(N), repeat = 2):
            if (r, c) in components: continue
            dfs(r, c)
            comp_id += 1
        for r, c in [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1)]:
            side_counts[components[(r, c)]] += 1
        for r in range(1, N):
            for c in range(N - 1):
                kernel = defaultdict(list)
                for dr in range(-1, 1):
                    for dc in range(2):
                        nr, nc = r + dr, c + dc
                        comp = components[(nr, nc)]
                        kernel[comp].append((nr, nc))
                for id, squares in kernel.items():
                    if len(squares) == 4: continue
                    if len(squares) in (1, 3): side_counts[id] += 1
                    if len(squares) == 2:
                        r1, c1, = squares[0]
                        r2, c2 = squares[1]
                        if r2 - r1 != 0 and c2 - c1 != 0:
                            side_counts[id] += 2
        for r, c in product(range(1, N - 1), repeat = 2):
            ans += side_counts[components[(r, c)]]
    print(ans)
calculate("big.txt")
```

## Day 13

### Solution 1:  system of linear equations, two equations, two unkowns, regex

1. Using the z3 solver can easily solve a system of linear equations.

```py
from z3.z3 import *

def solve_two_equations(a1, b1, a2, b2, c1, c2):
    x, y = Int("x"), Int("y")
    s = Solver()
    s.add(a1 * x + b1 * y == c1)
    s.add(a2 * x + b2 * y == c2)
    if s.check() == sat:
        model = s.model()
        x_val = model[x].as_long()
        y_val = model[y].as_long()
        return (x_val, y_val)
    return (-1, -1)

def calculate(filename):
    def extract(line):
        x, y = map(int, re.findall(pat, line))
        return x, y
    with open(filename, "r") as f:
        data = f.read().splitlines()
        pat = r"(\d+)"
        ax = ay = bx = by = tx = ty = ans = 0
        offset = 10000000000000
        for i in range(len(data)):
            if i % 4 == 0:
                ax, ay = extract(data[i])
            elif i % 4 == 1:
                bx, by = extract(data[i])
            elif i % 4 == 2:
                tx, ty = extract(data[i])
                x, y = solve_two_equations(ax, bx, ay, by, tx + offset, ty + offset)
                if x != -1: ans += 3 * x + y
        print(ans)
calculate("small.txt")
calculate("big.txt")
```

### Solution 2:  dynamic programming, cache

1. This solution is fun but it is O(n^2) time complexity per machine and it is not fast enough to handle when n is really large, such as part 2 where it is exceeding 10^9

```py
def calculate(filename):
    def extract(line):
        x, y = map(int, re.findall(pat, line))
        return x, y
    @cache 
    def dfs(x, y):
        if (x, y) == (0, 0): 
            return 0
        if x < 0 or y < 0: return math.inf
        res = min(dfs(x - ax, y - ay) + 3, dfs(x - bx, y - by) + 1)
        return res
    with open(filename, "r") as f:
        data = f.read().splitlines()
        pat = r"(\d+)"
        ax = ay = bx = by = tx = ty = ans = 0
        for i in range(len(data)):
            if i % 4 == 0:
                ax, ay = extract(data[i])
            elif i % 4 == 1:
                bx, by = extract(data[i])
            elif i % 4 == 2:
                tx, ty = extract(data[i])
                cost = dfs(tx, ty)
                dfs.cache_clear()
                if cost < math.inf: ans += cost
        print(ans)
calculate("small.txt")
calculate("big.txt")
```

## Day 14

### Solution 1:  approximate

1. Just convert to ascii art when there are some number of robots that are adjacent, that might look like christmas tree. 
1. Oh you only need maybe 20,000 iterations.  should work

```py
from PIL import Image, ImageDraw, ImageFont
def resize_image(input_path, output_path, scale_factor=0.5):
    """
    Resize the image by a scale factor to reduce its height and width proportionally.

    :param input_path: Path to the input image.
    :param output_path: Path to save the resized image.
    :param scale_factor: Scale factor to resize the image (e.g., 0.5 to reduce by 50%).
    """
    image = Image.open(input_path)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    resized_image.save(output_path, format="PNG")
    
def save_ascii_art_to_png(ascii_art, output_path, font_path=None, font_size=12, image_bg_color="black", text_color="white"):
    """
    Save ASCII art as a PNG image.

    :param ascii_art: The ASCII art string.
    :param output_path: The path where the PNG will be saved.
    :param font_path: Path to the .ttf font file (optional, defaults to system font).
    :param font_size: Font size for rendering the text.
    :param image_bg_color: Background color of the image.
    :param text_color: Color of the ASCII text.
    """
    # Split ASCII art into lines
    lines = ascii_art.splitlines()
    max_line_length = max(len(line) for line in lines)
    num_lines = len(lines)

    # Load a font (use default if font_path is None)
    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)



    # Calculate image dimensions using textbbox()
    dummy_image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_image)
    char_width = int(draw.textlength("A", font=font))  # Width of one character
    char_height = int(font.getbbox("A")[3])           # Height of one character
    line_spacing = 1
    # Adjust line height for compactness
    line_height = int(char_height * line_spacing)
    image_width = int(char_width * max_line_length)
    image_width = 80
    image_height = int(line_height * num_lines)

    # Create the image
    image = Image.new("RGB", (image_width, image_height), color=image_bg_color)
    draw = ImageDraw.Draw(image)

    # Draw the ASCII art onto the image
    for i, line in enumerate(lines):
        draw.text((0, i * char_height), line, fill=text_color, font=font)

    # Save the image as PNG
    image.save(output_path, format="PNG")
    print(f"ASCII art saved to {output_path}")
def calculate(filename):
    def get_grid(data):
        grid = [[" " for _ in range(C)] for _ in range(R)]
        for r, c in data:
            grid[r][c] = "#"
        return grid
    def display_data(grid):
        return "\n".join(["".join(row) for row in grid])
    def extract(line):
        x, y, w, z = map(int, re.findall(pat, line))
        return x, y, w, z
    def row_count(grid, threshold):
        cnt = 0
        for r, c in product(range(R), range(C)):
            if r == 0 or grid[r][c] != "#": cnt = 0
            if grid[r][c] == "#": cnt += 1
            if cnt > threshold: return True
        return False
    with open(filename, "r") as f:
        data = f.read().splitlines()
        pat = r"(-?\d+)"
        ans = 1
        R, C = 103, 101
        N = len(data)
        print("N", N)
        robots = []
        speed = []
        for line in data:
            c, r, dc, dr = extract(line)
            robots.append([r, c])
            speed.append((dr, dc))
        
        for i in range(10_000):
            grid = get_grid(robots)
            if row_count(grid, 5):
                ascii_art = display_data(grid)
                path = f"images/{i}.png"
                save_ascii_art_to_png(ascii_art, path, font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", font_size = 2)
            for i in range(N):
                robots[i][0] += speed[i][0]
                robots[i][0] %= R
                robots[i][1] += speed[i][1]
                robots[i][1] %= C
        print(ans)
calculate("big.txt")
```

## Day 15

### Solution 1: 

TODO

```py
DIRECTION = {
    "<": (0, -1),
    ">": (0, 1),
    "^": (-1, 0),
    "v": (1, 0)
}
MAP = {
    ".": "..",
    "@": "@.",
    "#": "##",
    "O": "[]"
}
def score(r, c):
    return 100 * r + c
def calculate(filename):
    def display_data(grid):
        return "\n".join(["".join(row) for row in grid])
    def in_bounds(r, c):
        return 0 <= r < R and 0 <= c < C
    def get_boxes(r, c, dr, dc):
        q = deque([(r, c)])
        boxes = set()
        while q:
            r, c = q.popleft()
            if i == 327:
                print(r, c)
            nr, nc = r + dr, c + dc
            if i == 327:
                print(nr, nc)
                print(point_to_box)
            if not in_bounds(nr, nc): continue
            if grid[nr][nc] == "#": return False, []
            if (nr, nc) in point_to_box:
                b = point_to_box[(nr, nc)]
                if b not in boxes:
                    boxes.add(b)
                    for rr, cc in box_to_point[b]:
                        q.append((rr, cc))
        return True, boxes
    def visualize(r, c):
        visgrid = [["."] * C for _ in range(R)]
        visgrid[r][c] = "@"
        for r, c in product(range(R), range(C)):
            if grid[r][c] == "#": visgrid[r][c] = "#"
        for b in range(len(box_to_point)):
            r1, c1 = box_to_point[b][0]
            r2, c2 = box_to_point[b][1]
            if c2 < c1:
                r1, r2 = r2, r1
                c1, c2 = c2, c1
            visgrid[r1][c1] = "["
            visgrid[r2][c2] = "]"
        return visgrid
    with open(filename, "r") as f:     
        instructions = []
        grid = []
        flag = False
        for line in f.read().splitlines():
            if not line:
                flag = True
                continue
            if flag:
                instructions.extend(line)
            else:
                row = "".join(map(lambda x: MAP[x], line))
                grid.append(list(row))
        R, C = len(grid), len(grid[0])
        box_to_point = defaultdict(list)
        point_to_box = {}
        for r, c in product(range(R), range(C)):
            if grid[r][c] == "]":
                b = len(box_to_point)
                box_to_point[b].extend([(r, c), (r, c - 1)])
                point_to_box[(r, c)] = point_to_box[(r, c - 1)] = b
        for r, c in product(range(R), range(C)):
            if grid[r][c] == "@":
                break
        for i, d in enumerate(instructions):
            # print(i, "dir", d, "r, c", r, c)
            dr, dc = DIRECTION[d]
            can_move, boxes = get_boxes(r, c, dr, dc)
            if not can_move: continue
            for b in boxes:
                points = copy.deepcopy(box_to_point[b])
                box_to_point[b].clear()
                # update points
                for rr, cc in points:
                    if point_to_box[(rr, cc)] == b:
                        del point_to_box[(rr, cc)]
                for rr, cc in points:
                    box_to_point[b].append((rr + dr, cc + dc))
                    point_to_box[(rr + dr, cc + dc)] = b
            r += dr 
            c += dc
            # print(display_data(visualize(r, c)))
        ans = 0
        visgrid = [["."] * C for _ in range(R)]
        visgrid[r][c] = "@"
        for r, c in product(range(R), range(C)):
            if grid[r][c] == "#": visgrid[r][c] = "#"
        for b in range(len(box_to_point)):
            r1, c1 = box_to_point[b][0]
            r2, c2 = box_to_point[b][1]
            if c2 < c1:
                r1, r2 = r2, r1
                c1, c2 = c2, c1
            visgrid[r1][c1] = "["
            visgrid[r2][c2] = "]"
            ans += score(r1, c1)
        print(ans)

calculate("small.txt")
calculate("big.txt")
```

## Day 16

### Solution 1:  grid, min heap, parent array, recover path, vectors, rotation

```py
DIRECTION = [(0, 1), (1, 0), (0, -1), (-1, 0)]
def calculate(filename):
    def in_bounds(r, c):
        return 0 <= r < N and 0 <= c < N
    with open(filename, "r") as f:
        grid = [list(row) for row in f.read().splitlines()]
        N = len(grid)
        minheap = []
        for r, c in product(range(N), repeat = 2):
            if grid[r][c] == "S":
                heapq.heappush(minheap, (0, r, c, 0))
                break
        dp = defaultdict(lambda: math.inf)
        parent = defaultdict(list)
        while minheap:
            cost, r, c, d = heapq.heappop(minheap)
            if grid[r][c] == "E":
                print("Part 1: ", cost)
                break
            for i in [-1, 1]:
                nd = (d + i) % 4
                ncost = cost + 1000
                if ncost <= dp[(r, c, nd)]:
                    if ncost < dp[(r, c, d)]:
                        parent[(r, c, nd)] = [(r, c, d)]
                    else:
                        parent[(r, c, nd)].append((r, c, d))
                    dp[(r, c, nd)] = ncost
                    heapq.heappush(minheap, (ncost, r, c, nd))
            dr, dc = DIRECTION[d]
            ncost = cost + 1
            nr = r + dr 
            nc = c + dc
            if not in_bounds(nr, nc) or grid[nr][nc] == "#": continue
            if ncost <= dp[(nr, nc, d)]:
                if ncost < dp[(nr, nc, d)]:
                    parent[(nr, nc, d)] = [(r, c, d)]
                else:
                    parent[(nr, nc, d)].append((r, c, d))
                dp[(nr, nc, d)] = ncost
                heapq.heappush(minheap, (ncost, nr, nc, d))
        vis = set()
        ans = set()
        stk = [(r, c, d)]
        while stk:
            r, c, d = stk.pop()
            ans.add((r, c))
            if (r, c, d) in vis: continue
            vis.add((r, c, d))
            if grid[r][c] == "S": continue
            for nr, nc, nd in parent[(r, c, d)]:
                stk.append((nr, nc, nd))
        print("Part 2: ", len(ans))
calculate("small.txt")
calculate("big.txt")
```

## Day 17

### Solution 1:  system of equations with bitwise operators, z3 solver,  bit-vector constraint problem

1. This problem if you analyze the input you can derive an equation which for mine was 
1. you need to solve this equation, f(x) = ((((x % 8) ^ 3) ^ floor(x / 2 ** (x % 8) ^ 3)) ^ 3) % 8
1. But then you have a system of equations because x bit shifted 3 times to the right, happens for each program index i
1. So there is a way to think of this in terms of octal, and a single octal shift. 
1. integer (or bit-vector) constraints. Such constraints are common in Satisfiability Modulo Theories (SMT) problems, which Z3 is designed to handle.
1. complex integer/bit-vector constraint satisfaction problem

TODO

```py
from z3 import *

def parse_input(filename):
    with open(filename, "r") as f:
        data = f.read().splitlines()
        registers, program = {}, []
        for line in data:
            if "Register" in line:
                name, value = line.split(":")
                registers[name[-1]] = int(value)
            elif "Program" in line:
                _, values = line.split(":")
                program = list(map(int, values.split(",")))
    return registers, program

def f(x):
    mod8 = x & 7
    pow_val = mod8 ^ 3 
    div_part = If(pow_val == 0, LShR(x,0),
              If(pow_val == 1, LShR(x,1),
              If(pow_val == 2, LShR(x,2),
              If(pow_val == 3, LShR(x,3),
              If(pow_val == 4, LShR(x,4),
              If(pow_val == 5, LShR(x,5),
              If(pow_val == 6, LShR(x,6),
                 LShR(x,7))))))))
    step1 = mod8 ^ 3
    step2 = step1 ^ div_part 
    step3 = step2 ^ 3 
    ans = step3 & 7
    return ans

def calculate(filename):
    registers, program = parse_input(filename)
    x = BitVec("x", 64)
    solver = Solver()
    solver.add(x >= (1 << 45))
    solver.add(x < (1 << 48))
    for i, val in enumerate(program):
        shifted = LShR(x, 3 * i)
        solver.add(f(shifted) == val)
    if solver.check() == sat:
        m = solver.model()
        solution = m[x].as_long()
        print(f"Solution found: x = {solution}")
    else:
        print("No solution found")
```

```py
def parse_input(filename):
    with open(filename, "r") as f:
        data = f.read().splitlines()
        registers, program = {}, []
        for line in data:
            if "Register" in line:
                name, value = line.split(":")
                registers[name[-1]] = int(value)
            elif "Program" in line:
                _, values = line.split(":")
                program = list(map(int, values.split(",")))
    return registers, program
def calculate(filename):
    def combo_operand(operand):
        if operand <= 3: return operand
        if operand == 4: return registers["A"]
        if operand == 5: return registers["B"]
        if operand == 6: return registers["C"]
        return 0
    def evaluate(pointer, opcode, operand):
        if opcode == 0:
            val = registers["A"] // 2 ** combo_operand(operand)
            registers["A"] = val
        elif opcode == 1:
            val = registers["B"] ^ operand
            registers["B"] = val
        elif opcode == 2:
            val = combo_operand(operand) % 8
            registers["B"] = val
        elif opcode == 3:
            if registers["A"] > 0:
                return -1, operand
        elif opcode == 4:
            val = registers["B"] ^ registers["C"] 
            registers["B"] = val
        elif opcode == 5:
            return combo_operand(operand) % 8, pointer + 2
        elif opcode == 6:
            val = registers["A"] // 2 ** combo_operand(operand)
            registers["B"] = val
        else:
            val = registers["A"] // 2 ** combo_operand(operand)
            registers["C"] = val
        return -1, pointer + 2
    def matches_program(val):
        registers["A"] = val
        registers["B"] = registers["C"] = 0
        idx = pointer = 0
        path = []
        # counter = Counter()
        while pointer < len(program):
            # counter[pointer] += 1
            path.append(pointer)
            output, pointer = evaluate(pointer, program[pointer], program[pointer + 1])
            # if output != -1:
            #     if idx == len(program) or output != program[idx]: return False
            #     idx += 1
        # print(val, counter)
        # print(val, path)
        return idx == len(program)
    registers, program = parse_input(filename)
    for v in range(1, 10 ** 5):
        if matches_program(v):
            print(v)
            break

# calculate("small.txt")
calculate("big.txt")
```

## Day 18

### Solution 1:  bfs, queue, grid, binary search

```py
def calculate(filename):
    def neighborhood(r, c):
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    def in_bounds(r, c):
        return 0 <= r < N and 0 <= c < N
    def possible(target):
        grid = [["."] * N for _ in range(N)]
        for i in range(target):
            r, c = points[i]
            grid[r][c] = "#"
        queue = deque([(0, 0)])
        vis = set()
        while queue:
            r, c = queue.popleft()
            if (r, c) == (N - 1, N - 1): 
                return True
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc) or (nr, nc) in vis or grid[nr][nc] == "#": continue 
                queue.append((nr, nc))
                vis.add((nr, nc))
        return False
    N = 71
    with open(filename, "r") as f:
        data = f.read().splitlines()
        points = []
        for line in data:
            r, c = map(int, line.split(","))
            points.append((r, c))
    lo, hi = 0, len(data)
    while lo < hi:
        mi = (lo + hi + 1) >> 1
        if possible(mi): lo = mi
        else: hi = mi - 1
    print(*points[lo])
calculate("big.txt")
```

## Day 19

### Solution 1:  dynamic programming, string

1. take each word, if it exists in items and dp[k] > 0, then you can say that it is 
1. O(n^2) per each word.
1. Yeah if I wanted to get it to actually be O(N^2) basically I'd have to use a trie data structure.  

```py
def calculate(filename):
    with open(filename, "r") as f:
        data = f.read().splitlines()
        ans = 0
        items = set(map(lambda x: x.strip(), data[0].split(",")))
        for i in range(2, len(data)):
            word = data[i]
            N = len(word)
            dp = [0] * (N + 1)
            dp[0] = 1
            for j in range(N + 1):
                for k in range(j - 1, -1, -1):
                    s = word[k : j]
                    if s in items and dp[k] > 0:
                        dp[j] += dp[k]
            ans += dp[N]
    print(ans)
calculate("big.txt")
```

## Day 20

### Solution 1:  bfs, memoization

```py
def calculate(filename):
    def neighborhood(r, c):
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    def in_bounds(r, c):
        return 0 <= r < N and 0 <= c < N
    def explore(r, c):
        dp = [[math.inf] * N for _ in range(N)]
        q = deque([(r, c)])
        steps = 0
        while q:
            for _ in range(len(q)):
                r, c = q.popleft()
                dp[r][c] = steps
                if grid[r][c] == "E": return dp
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or grid[nr][nc] == "#" or dp[nr][nc] != math.inf: continue 
                    q.append((nr, nc))
            steps += 1
        return dp
    def bfs(r, c, threshold):
        q = deque([(r, c)])
        start_time = st[r][c]
        step = res = 0
        vis = set([(r, c)])
        while q and step <= threshold:
            for _ in range(len(q)):
                r, c = q.popleft()
                if grid[r][c] != "#":
                    saved_time = st[r][c] - start_time - step
                    if saved_time >= 100: res += 1
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or (nr, nc) in vis: continue
                    vis.add((nr, nc))
                    q.append((nr, nc))
            step += 1
        return res
    with open(filename, "r") as f:
        grid = [list(row) for row in f.read().splitlines()]
        ans = 0
        N = len(grid)
        T = 20 # T = 2 is answer for part 1
        st = None
        for r, c in product(range(N), repeat = 2):
            if grid[r][c] == "S":
                st = explore(r, c)
        for r, c in product(range(N), repeat = 2):
            if grid[r][c] == "#": continue
            ans += bfs(r, c, T)
    print(ans)
calculate("big.txt")
```

### Solution 2: manhattan distance, looping

1. looping over points in the maze paths instead
1. The cheat mode is activated to get from the start to the end point is the manhattan distance, that is the shortest way to cheat and get from point a to point b.  So it saves the most time. 

TODO

```py
def calculate(filename):
    def neighborhood(r, c):
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    def in_bounds(r, c):
        return 0 <= r < N and 0 <= c < N
    def bfs(r, c):
        dp = [[math.inf] * N for _ in range(N)]
        q = deque([(r, c)])
        steps = 0
        while q:
            for _ in range(len(q)):
                r, c = q.popleft()
                dp[r][c] = steps
                if grid[r][c] == "E": return dp
                for nr, nc in neighborhood(r, c):
                    if not in_bounds(nr, nc) or grid[nr][nc] == "#" or dp[nr][nc] != math.inf: continue 
                    q.append((nr, nc))
            steps += 1
        return dp
    def manhattan_distance(r1, c1, r2, c2):
        return abs(r2 - r1) + abs(c2 - c1)
    with open(filename, "r") as f:
        grid = [list(row) for row in f.read().splitlines()]
        ans = 0
        N = len(grid)
        st = None
        points = []
        for r, c in product(range(N), repeat = 2):
            if grid[r][c] == "S":
                st = bfs(r, c)
            if grid[r][c] != "#":
                points.append((r, c))
        for i in range(len(points)):
            for j in range(i):
                r1, c1 = points[i]
                r2, c2 = points[j]
                dist = manhattan_distance(r1, c1, r2, c2)
                if dist > 20: continue
                delta = abs(st[r2][c2] - st[r1][c1]) - dist
                if delta >= 100: ans += 1
    print(ans)
calculate("big.txt")
```

## Day 21

### Solution 1:  combinatorics, shortest path, recursion, backtracking, dynamic programming

1. Figuring out that the sequence of paths is always starting from A and ending on A, is needed.
1. Figuring out that sequence of paths other than A for direction keypads will consist of only one horizontal and one vertical movements.
1. solve for depth, and recurse and have depth = 0 be special that returns the length of sequence needed. 

```py
DIRECTION = {
    ">": (0, 1),
    "v": (1, 0),
    "<": (0, -1),
    "^": (-1, 0)
}

DIRECTION_KEYPAD = {
    "^": (0, 1),
    "v": (1, 1),
    "<": (1, 0),
    ">": (1, 2),
    "A": (0, 2)
}

NUMERIC_KEYPAD = {
    "7": (0, 0),
    "8": (0, 1),
    "9": (0, 2),
    "4": (1, 0),
    "5": (1, 1),
    "6": (1, 2),
    "1": (2, 0),
    "2": (2, 1),
    "3": (2, 2),
    "0": (3, 1),
    "A": (3, 2) 
}

 
def generate_combinations(char_a, count_a, char_b, count_b):
    """ 
    Generate all combinations of a string containing `count_a` of `char_a` and 
    `count_b` of `char_b`, arranged in every possible order.
    """
    for idx in combinations(range(count_a + count_b), r = count_a):
        res = [char_b] * (count_a + count_b)
        for i in idx: res[i] = char_a
        yield "".join(res)

@cache
def generate_ways(a, b, is_direction):
    keypad = DIRECTION_KEYPAD if is_direction else NUMERIC_KEYPAD
    cur_loc = keypad[a]
    next_loc = keypad[b]
    dr = next_loc[0] - cur_loc[0]
    dc = next_loc[1] - cur_loc[1]
    moves = []
    if dr > 0:
        moves.extend(["v", dr])
    else:
        moves.extend(["^", -dr])
    if dc > 0:
        moves.extend([">", dc])
    else:
        moves.extend(["<", -dc])
    raw_combos = [x + "A" for x in generate_combinations(*moves)]
    combos = []
    for combo in raw_combos:
        r, c = cur_loc 
        is_valid = True
        for ch in combo[:-1]:
            dr, dc = DIRECTION[ch]
            r, c = r + dr, c + dc
            if (r, c) not in keypad.values():
                is_valid = False
                break
        if is_valid:
            combos.append(combo)
    return combos

@cache
def get_cost(a, b, is_direction, depth = 0):
    if depth == 0:
        return len(min(generate_ways(a, b, True), key = len))
    ways = generate_ways(a, b, is_direction)
    best_cost = math.inf
    for seq in ways:
        seq = "A" + seq # always starting from A
        cost = 0
        for i in range(1, len(seq)):
            a, b = seq[i - 1], seq[i]
            cost += get_cost(a, b, True, depth - 1)
        best_cost = min(best_cost, cost)
    return best_cost

def get_code_cost(code, depth):
    code = "A" + code 
    cost = 0
    for i in range(1, len(code)):
        a, b = code[i - 1], code[i]
        cost += get_cost(a, b, False, depth)
    return cost


def calculate(filename):
    with open(filename, "r") as f:
        data = f.read().splitlines()
    ans = 0
    for code in data:
        ans += get_code_cost(code, 25) * int(code[:-1])
    print(ans)

calculate("big.txt")
```

## Day 22

### Solution 1:  precomputation, list of maps, deque

1. Preprocess all the possible change sequences of length 4, there are roughly 40,000 possible candidates. 
1. Precompute the change sequences of length 4 using a deque of length 4 always, and placing into a dictionary for each integer.
1. So you compute the price which is last digit for each secret number with the change sequence matching

```py
def calculate(filename):
    mod = 16777216
    with open(filename, "r") as f:
        data = list(map(int, f.read().splitlines()))
    N = len(data)
    dp = [{} for _ in range(N)]
    for i, x in enumerate(data):
        q = deque()
        for _ in range(2_000):
            d = x % 10
            x ^= x * 64
            x %= mod
            x ^= x // 32
            x %= mod
            x ^= x * 2048
            x %= mod
            nd = x % 10
            delta = nd - d
            q.append(delta)
            if len(q) == 4:
                state = tuple(q)
                if state not in dp[i]:
                    dp[i][state] = nd
                q.popleft()
    vis = set()
    for i, j, k, w, z in product(range(10), repeat = 5):
        diff = (j - i, k - j, w - k, z - w)
        vis.add(diff)
    ans = 0
    for state in vis:
        cand = 0
        for i in range(N):
            cand += dp[i].get(state, 0)
        ans = max(ans, cand)
    print(ans)
    
calculate("small.txt")
calculate("big.txt")
```

## Day 23

### Solution 1:  bron-kerbosch algorithm, clique

1. Modify bron-kerbosch algorithm so it is not returning maximal cliques, but all cliques of size 3. 
1. You have to remove the pivot optimization for this to work.  

```py
def calculate(filename):
    def bron_kerbosch(R, P, X):
        if len(R) == 3: 
            yield R
            return
        if not P and not X: yield R
        for v in P.copy():
            yield from bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    adj = defaultdict(set)
    with open(filename, "r") as f:
        data = f.read().splitlines()
        for line in data:
            u, v = line.split("-")
            adj[u].add(v)
            adj[v].add(u)
    ans = 0
    for clique in bron_kerbosch(set(), set(adj), set()):
        if len(clique) == 3 and any(c[0] == "t" for c in clique): ans += 1
    print(ans)
    
calculate("small.txt")
calculate("big.txt")
```

### Solution 2:  undirected graph, maximum clique, networkx library

1. Can use the networkx find_cliques to find all maximal cliques. 

```py
def calculate(filename):
    adj = defaultdict(set)
    with open(filename, "r") as f:
        data = f.read().splitlines()
        for line in data:
            u, v = line.split("-")
            adj[u].add(v)
            adj[v].add(u)
    G = nx.Graph()
    for u, v in adj.items():
        for w in v:
            G.add_edge(u, w)
    largest_clique = max(nx.find_cliques(G), key=len)
    ans = ",".join(sorted(largest_clique))
    print(ans)
    
calculate("small.txt")
```

### Solution 3:  Bron-Kerbosch algorithm, undirected graph, maximum clique

```py
def calculate(filename):
    def bron_kerbosch(R, P, X):
        if not P and not X: yield R
        pivot = next(iter(P | X)) if P or X else None
        non_neighbors = P - adj[pivot] if pivot else P

        for v in non_neighbors:
            yield from bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    adj = defaultdict(set)
    with open(filename, "r") as f:
        data = f.read().splitlines()
        for line in data:
            u, v = line.split("-")
            adj[u].add(v)
            adj[v].add(u)
    cliques = list(bron_kerbosch(set(), set(adj), set()))
    largest_clique = max(cliques, key=len)
    ans = ",".join(sorted(largest_clique))
    print(ans)
    
calculate("small.txt")
```

### Solution 4:  recursion, dfs, set, largest clique, set theory, subset

```py
def calculate(filename):
    def dfs(u, clique):
        elem = tuple(sorted(clique))
        if elem in cliques: return
        cliques.add(elem)
        for v in adj[u]:
            if clique <= adj[v]:
                dfs(v, clique | {v})
    adj = defaultdict(set)
    with open(filename, "r") as f:
        data = f.read().splitlines()
        for line in data:
            u, v = line.split("-")
            adj[u].add(v)
            adj[v].add(u)
    cliques = set()
    for u in adj:
        dfs(u, set())
    largest_clique = max(cliques, key = len)
    ans = ",".join(sorted(largest_clique))
    print(ans)
calculate("small.txt")
calculate("big.txt")
```

## Day 24

### Solution 1:  directed graph, topological sort, sorting pairs or gates, bitwise operations

```py
def calculate(filename):
    adj = defaultdict(list)
    deg = Counter()
    values = {}
    nodes = set()
    operation = defaultdict(str)
    with open(filename, "r") as f:
        data = f.read().splitlines()
        flag = False
        for line in data:
            if not line:
                flag = True
                continue
            if flag:
                if "XOR" in line:
                    prefix, output = line.split(" -> ")
                    deg[output] += 2
                    u, v = prefix.split(" XOR ")
                    nodes |= {u, v, output}
                    adj[u].append(output)
                    adj[v].append(output)
                    operation[(u, v, output)] = "XOR"
                elif "AND" in line:
                    prefix, output = line.split(" -> ")
                    deg[output] += 2
                    u, v = prefix.split(" AND ")
                    adj[u].append(output)
                    adj[v].append(output)
                    nodes |= {u, v, output}
                    operation[(u, v, output)] = "AND"
                elif "OR" in line:
                    prefix, output = line.split(" -> ")
                    deg[output] += 2
                    u, v = prefix.split(" OR ")
                    adj[u].append(output)
                    adj[v].append(output)
                    nodes |= {u, v, output}
                    operation[(u, v, output)] = "OR"
            else:
                u, v = line.split(": ")
                values[u] = int(v)
    top_sort = []
    dq = deque()
    for u in nodes:
        if deg[u] == 0: dq.append(u)
    vis = set()
    while dq:
        u = dq.popleft()
        if u in vis: continue 
        vis.add(u)
        top_sort.append(u)
        for v in adj[u]:
            deg[v] -= 1
            if deg[v] == 0:
                dq.append(v)
    gates = sorted(operation, key = lambda x: max(top_sort.index(x[0]), top_sort.index(x[1])))
    for u, v, w in gates:
        if operation[(u, v, w)] == "AND":
            values[w] = values[u] & values[v]
        elif operation[(u, v, w)] == "OR":
            values[w] = values[u] | values[v]
        elif operation[(u, v, w)] == "XOR":
            values[w] = values[u] ^ values[v]
    nodes = sorted(filter(lambda x: x.startswith("z"), values), reverse = True)
    ans = int("".join(str(values[x]) for x in nodes), 2)
    print(ans)
    
calculate("small.txt")
calculate("big.txt")
```

### Solution 2:  Arithmetic Logic Unit (ALU), logic gates, bitwise operations, directed graph, full adder

1. The way I solved this was a little non-conventional possibly and requires visualizing the graph using graphviz. 
1. So it becomes obvious that it is composed of full adders, and one half adder, but this is for adding each bit using logic gates, to get the output bit and carry bit
1. I wrote a bit sloppily the code that finds where the full adder is wrong, then looking at the graph I can identify what needs to be switched. 
1. And you fix this in order from the least significant bit, and add in the swap and then you find the second, until you've found the fourth.
1. I wonder if there is a pure programmatic way to solve this that could scale to many more bits, cause this is only reasonable because only 4 swaps required.  This is a further challenge for me. 

half adder
xor(x, y) = z
and(x, y) = c_out

full adder
xor(xor(x, y), c_in) = z
or(and(c_in, xor(x, y)), and(x, y)) = c_out

```py
def calculate(filename):
    adj = defaultdict(list)
    deg = Counter()
    values = {}
    nodes = set()
    operation = defaultdict(list)
    dot = graphviz.Digraph()
    swaps = {}
    swaps["qnw"] = "qff"
    swaps["qff"] = "qnw"
    swaps["pbv"] = "z16"
    swaps["z16"] = "pbv"
    swaps["z23"] = "qqp"
    swaps["qqp"] = "z23"
    swaps["z36"] = "fbq"
    swaps["fbq"] = "z36"
    with open(filename, "r") as f:
        data = f.read().splitlines()
        flag = False
        for line in data:
            if not line:
                flag = True
                continue
            if flag:
                prefix, w = line.split(" -> ")
                w = swaps.get(w, w)
                deg[w] += 2
                if "XOR" in line:
                    u, v = prefix.split(" XOR ")
                    if u > v: u, v = v, u
                    nodes |= {u, v, w}
                    adj[u].append(w)
                    adj[v].append(w)
                    dot.edge(u, w, label = "XOR")
                    dot.edge(v, w, label = "XOR")
                    operation[(u, v)].append((w, "XOR"))
                elif "AND" in line:
                    u, v = prefix.split(" AND ")
                    if u > v: u, v = v, u
                    adj[u].append(w)
                    adj[v].append(w)
                    dot.edge(u, w, label = "AND")
                    dot.edge(v, w, label = "AND")
                    nodes |= {u, v, w}
                    operation[(u, v)].append((w, "AND"))
                elif "OR" in line:
                    u, v = prefix.split(" OR ")
                    if u > v: u, v = v, u
                    adj[u].append(w)
                    adj[v].append(w)
                    dot.edge(u, w, label = "OR")
                    dot.edge(v, w, label = "OR")
                    nodes |= {u, v, w}
                    operation[(u, v)].append((w, "OR"))
            else:
                u, v = line.split(": ")
                values[u] = int(v)
    top_sort = []
    dq = deque()
    for u in nodes:
        if deg[u] == 0: 
            dq.append(u)
    vis = set()
    while dq:
        u = dq.popleft()
        if u in vis: continue
        vis.add(u)
        top_sort.append(u)
        for v in adj[u]:
            deg[v] -= 1
            if deg[v] == 0:
                dq.append(v)
    gates = sorted(operation, key = lambda x: max(top_sort.index(x[0]), top_sort.index(x[1])))
    for u, v in gates:
        for w, op in operation[(u, v)]:
            if op == "AND":
                values[w] = values[u] & values[v]
            elif op == "OR":
                values[w] = values[u] | values[v]
            elif op == "XOR":
                values[w] = values[u] ^ values[v]
    X = sorted(filter(lambda x: x.startswith("x"), values), reverse = True)
    Y = sorted(filter(lambda x: x.startswith("y"), values), reverse = True)
    Z = sorted(filter(lambda x: x.startswith("z"), values), reverse = True)
    x = int("".join(str(values[x]) for x in X), 2)
    y = int("".join(str(values[x]) for x in Y), 2)
    # swap 4 pairs of outputs so that z = x + y
    z = int("".join(str(values[x]) for x in Z), 2)
    z_bin = list(reversed(bin(z)[2:]))
    expected = list(reversed(bin(x + y)[2:]))
    for i in range(46):
        if z_bin[i] != expected[i]:
            print(i)
    output_path = dot.render(
        filename='graph', 
        format='png',
        directory='output',
        cleanup=True
    )
    carry = first = second = third = None
    for w, op in operation[("x00", "y00")]:
        if op == "AND": carry = w
    print("carry", carry)
    for i in range(1, 46):
        u, v = f"x{i:02}", f"y{i:02}"
        for w, op in operation[(u, v)]:
            if op == "XOR":
                first = w
            elif op == "AND":
                second = w
            else:
                print(u, v, w, op)
        cnt = 0
        for w, op in operation[(min(first, carry), max(first, carry))]:
            cnt += 1
            if op == "AND":
                third = w
            elif op == "XOR":
                if w != f"z{i:02}":
                    print(first, carry, w, op)
        if cnt == 0: print("first, and carry", first, carry)
        cnt = 0
        for w, op in operation[(min(second, third), max(second, third))]:
            cnt += 1
            if op == "OR":
                carry = w
            else:
                print(second, third, w, op)
        if cnt == 0: print("second and third", second, third)
    print(z)
calculate("big.txt")
```

## Day 25

### Solution 1:  grid, convert grid to vector

```py
def is_lock(grid):
    return all(grid[0][i] == "#" for i in range(len(grid[0])))
def size(grid):
    res = [0] * len(grid[0])
    R, C = len(grid), len(grid[0])
    for r, c in product(range(R), range(C)):
        if grid[r][c] == "#":
            res[c] += 1
    return res
def calculate(filename):
    keys_and_locks = []
    with open(filename, "r") as f:
        data = f.read().splitlines() + [""]
        grid = []
        for line in data:
            if not line:
                keys_and_locks.append(grid.copy())
                grid = []
            else:
                grid.append(list(line))
    keys, locks = [], []
    for grid in keys_and_locks:
        if is_lock(grid):
            locks.append(size(grid))
        else:
            keys.append(size(grid))
    ans = 0
    for key, lock in product(keys, locks):
        if all(key[i] + lock[i] <= 7 for i in range(len(key))): ans += 1
    print(ans)

calculate("big.txt")
```