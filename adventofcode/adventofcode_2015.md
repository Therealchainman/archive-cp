# Advent of Code 2015

## Day 1: 

### Part 2

```py

```

## Day 2: 

### Part 2

```py

```

## Day 3: 

### Part 2

```py

```

## Day 4: 

### Part 1

what you need to know
1. 1 byte = 2 hex digits
1. digest is 16 bytes or 32 hex digits
1. byte = 8 bits, and you need 4 bits for a hex digit.

digest[0] == 0 &&        // hex[0] and hex[1] are 0
digest[1] == 0 &&        // hex[2] and hex[3] are 0
(digest[2] & 0xF0) == 0  // hex[4] is 0

0xF0 in binary is 1111 0000.
0x prefix means writing in hexadecimal (base 16)
So when you do & operations you are checking that the hex[4] all the bits are 0s. 

why can't you do cout << digest[0], 

digest[0] is an unsigned char, so cout treats it like a character, not a number.
If the value is, say, 0x07 or 0xB3, that is not a printable ASCII character. You might see:
Nothing
Gibberish
Terminal beeps or weird symbols

```cpp
const int MAXN = 1e7;

void solve() {
    unsigned char digest[MD5_DIGEST_LENGTH];
    string s;
    cin >> s;
    for (int i = 0; i < MAXN; ++i) {
        string cand = s + to_string(i);
        MD5(reinterpret_cast<const unsigned char*>(cand.c_str()), cand.size(), digest);
        if (digest[0] == 0 && digest[1] == 0 && (digest[2] & 0xF0) == 0) {
            debug(i, "\n");
            return;
        }
    }
}
```

### Part 2

1. find hex that starts with six zeros. 

```cpp
const int MAXN = 1e7;

void solve() {
    unsigned char digest[MD5_DIGEST_LENGTH];
    string s;
    cin >> s;
    for (int i = 0; i < MAXN; ++i) {
        string cand = s + to_string(i);
        MD5(reinterpret_cast<const unsigned char*>(cand.c_str()), cand.size(), digest);
        if (digest[0] == 0 && digest[1] == 0 && digest[2] == 0) {
            debug(i, "\n");
            return;
        }
    }
}
```

## Day 5: 

### Part 1: set, string, boolean

```cpp
unordered_set<string> banned = {
    "ab", "cd", "pq", "xy"
};

bool isVowel(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

bool isBanned(char a, char b) {
    string s = "";
    s += a;
    s += b;
    return banned.find(s) != banned.end();
}

bool isNice(const string& s) {
    int N = s.size();
    int vowelCount = 0;
    bool hasDouble = false, hasBanned = false;
    for (int i = 0; i < N; ++i) {
        if (isVowel(s[i])) vowelCount++;
        if (i > 0 && s[i] == s[i - 1]) hasDouble = true;
        if (i > 0 && isBanned(s[i - 1], s[i])) hasBanned = true;
    }
    return vowelCount >= 3 && hasDouble && !hasBanned;
}

void solve() {
    int ans = 0;
    string s;
    while (getline(cin, s)) {
        if (isNice(s)) ans++;
    }
    debug(ans, "\n");
}
```

### Part 2: string, set, boolean

```cpp
bool isXYX(char a, char b, char c) {
    return a == c;
}

bool isNice(const string& s) {
    int N = s.size();
    bool hasXYX = false, hasPair = false;
    set<string> pairs;
    string lastPair = "";
    for (int i = 0; i < N; ++i) {
        if (i > 1 && isXYX(s[i - 2], s[i - 1], s[i])) {
            hasXYX = true;
        }
        if (i > 0) {
            string curPair = s.substr(i - 1, 2);
            if (pairs.find(curPair) != pairs.end()) {
                hasPair = true;
            }
            if (!lastPair.empty()) {
                pairs.insert(lastPair);
            }
            lastPair = curPair;
        }
    }
    return hasXYX && hasPair;
}

void solve() {
    int ans = 0;
    string s;
    while (getline(cin, s)) {
        if (isNice(s)) ans++;
    }
    debug(ans, "\n");
}
```

## Day 6: 

### Part 1: compile pattern that was fixed, and grid of lights

```py
turn_on = compile("turn on {:d},{:d} through {:d},{:d}")
turn_off = compile("turn off {:d},{:d} through {:d},{:d}")
toggle = compile("toggle {:d},{:d} through {:d},{:d}")
with open('input.txt', 'r') as f:
    data = f.read().splitlines()
    lights = [[0] * 1000 for _ in range(1000)]
    for line in data:
        to, toff, tog = turn_on.parse(line), turn_off.parse(line), toggle.parse(line)
        if to is not None:
            r1, c1, r2, c2 = to.fixed
            for r, c in product(range(r1, r2 + 1), range(c1, c2 + 1)):
                lights[r][c] = 1
        elif toff is not None:
            r1, c1, r2, c2 = toff.fixed
            for r, c in product(range(r1, r2 + 1), range(c1, c2 + 1)):
                lights[r][c] = 0
        else:
            r1, c1, r2, c2 = tog.fixed
            for r, c in product(range(r1, r2 + 1), range(c1, c2 + 1)):
                lights[r][c] ^= 1
    print(sum(map(sum, lights)))
```

## Day 6:

### Part 2: 

```py
turn_on = compile("turn on {:d},{:d} through {:d},{:d}")
turn_off = compile("turn off {:d},{:d} through {:d},{:d}")
toggle = compile("toggle {:d},{:d} through {:d},{:d}")
with open('input.txt', 'r') as f:
    data = f.read().splitlines()
    lights = [[0] * 1000 for _ in range(1000)]
    for line in data:
        to, toff, tog = turn_on.parse(line), turn_off.parse(line), toggle.parse(line)
        if to is not None:
            r1, c1, r2, c2 = to.fixed
            for r, c in product(range(r1, r2 + 1), range(c1, c2 + 1)):
                lights[r][c] += 1
        elif toff is not None:
            r1, c1, r2, c2 = toff.fixed
            for r, c in product(range(r1, r2 + 1), range(c1, c2 + 1)):
                lights[r][c] = max(0, lights[r][c] - 1)
        else:
            r1, c1, r2, c2 = tog.fixed
            for r, c in product(range(r1, r2 + 1), range(c1, c2 + 1)):
                lights[r][c] += 2
    print(sum(map(sum, lights)))
```

## Day 7: 

### Part 2:  bitwise operations, directed graph, topological sort, don't process a wire till all of it's prerequisites have been visited

The bitwise NOT, or bitwise complement, is a unary operation that performs logical negation on each bit, forming the ones' complement of the given binary value. Bits that are 0 become 1, and those that are 1 become 0.

To perform the NOT bitwise operator in python is a little tricky.  The not operator should flip all the bits, so given in this problem you have from to 0 to 65535, so it is an unsigned integer.  Since this is a 16 bit integer, you can just use xor to flip the bits and perform the not operator by xor the integer with 65535.

```py
operation = {}
adj = defaultdict(set)
def preprocess(data):
    indegrees = Counter()
    for line in data:
        if "AND" in line:
            operands, w = line.split(" -> ")
            v1, v2 = operands.split(" AND ")
            if not v1.isdigit(): indegrees[w] += 1
            if not v2.isdigit(): indegrees[w] += 1
        elif "OR" in line:
            operands, w = line.split(" -> ")
            v1, v2 = operands.split(" OR ")
            if not v1.isdigit(): indegrees[w] += 1
            if not v2.isdigit(): indegrees[w] += 1
            wires[w] = wires[v1] | wires[v2]
        elif "LSHIFT" in line:
            operands, w = line.split(" -> ")
            v, shift = operands.split(" LSHIFT ")
            if not v.isdigit(): indegrees[w] += 1
        elif "RSHIFT" in line:
            operands, w = line.split(" -> ")
            v, shift = operands.split(" RSHIFT ")
            if not v.isdigit(): indegrees[w] += 1
        elif "NOT" in line:
            operands, w = line.split(" -> ")
            v = operands.split("NOT ")[1]
            if not v.isdigit(): indegrees[w] += 1
        else:
            v, w = line.split(" -> ")
            if not v.isdigit(): indegrees[w] += 1
    return indegrees
def set_operations(data):
    for line in data:
        if "AND" in line:
            operands, w = line.split(" -> ")
            v1, v2 = operands.split(" AND ")
            if not v1.isdigit(): adj[v1].add(w)
            if not v2.isdigit(): adj[v2].add(w)
            operation[w] = (v1, v2, "AND")
        elif "OR" in line:
            operands, w = line.split(" -> ")
            v1, v2 = operands.split(" OR ")
            if not v1.isdigit(): adj[v1].add(w)
            if not v2.isdigit(): adj[v2].add(w)
            operation[w] = (v1, v2, "OR")
            wires[w] = wires[v1] | wires[v2]
        elif "LSHIFT" in line:
            operands, w = line.split(" -> ")
            v, shift = operands.split(" LSHIFT ")
            if not v.isdigit(): adj[v].add(w)
            operation[w] = (v, shift, "LSHIFT")
        elif "RSHIFT" in line:
            operands, w = line.split(" -> ")
            v, shift = operands.split(" RSHIFT ")
            if not v.isdigit(): adj[v].add(w)
            operation[w] = (v, shift, "RSHIFT")
        elif "NOT" in line:
            operands, w = line.split(" -> ")
            v = operands.split("NOT ")[1]
            if not v.isdigit(): adj[v].add(w)
            operation[w] = (v, "0", "NOT")
        else:
            v, w = line.split(" -> ")
            if not v.isdigit(): adj[v].add(w)
            operation[w] = (v, "0", "ASSIGN")
def apply_operation(v1, v2, op):
    if op == "AND":
        wires[w] = v1 & v2
    elif op == "OR":
        wires[w] = v1 | v2
    elif op == "NOT":
        wires[w] = mask ^ v1
    elif op == "LSHIFT":
        wires[w] = v1 << v2
    elif op == "RSHIFT":
        wires[w] = v1 >> v2
    else:
        wires[w] = v1
with open('input.txt', 'r') as f:
    data = f.read().splitlines()
    mask = 65535
    set_operations(data)
    for i in range(2):
        if i > 0: # override the assign operation to wire b based on the value of wire a
            operation["b"] = (str(wires["a"]), "0", "ASSIGN")
        wires = Counter()
        indegrees = preprocess(data)
        queue = deque()
        for w in operation.keys():
            if indegrees[w] == 0: queue.append(w)
        while queue:
            w = queue.popleft()
            v1, v2, op = operation[w]
            v1 = int(v1) if v1.isdigit() else wires[v1]
            v2 = int(v2) if v2.isdigit() else wires[v2]
            apply_operation(v1, v2, op)
            for v in adj[w]:
                indegrees[v] -= 1
                if indegrees[v] == 0: queue.append(v)
    print(wires["a"])
```

## Day 8: 

### Part 1:  regex, string replacement, escape character of strings

Normally, Python uses backslashes as escape characters. Prefacing the string definition with 'r' is a useful way to define a string where you need the backslash to be an actual backslash and not part of an escape code that means something else in the string.

In Python, the backslash \ is an escape character in string literals. When you use r"\", the r prefix denotes a raw string, but it doesn't suppress the escape behavior of the backslash within the string literal. So, r"\" is considered an invalid raw string because it ends with an escape character without a corresponding character to escape.

```py
with open('input.txt', 'r') as f:
    data = f.read().splitlines()
    cnt = total = 0
    pattern = re.compile(r"\\x[0-9a-f]{2}")
    for line in data:
        total += len(line)
        line = re.sub(pattern, "|", line)
        line = line.replace(r'\"', '"').replace(r"\\", "\\")
        cnt += len(line) - 2
    print(total - cnt)
```

## Day 8: 

### Part 2:  make translate table and use str.maketrans and str.translate

```py
with open('input.txt', 'r') as f:
    data = f.read().splitlines()
    cnt = total = 0
    map_table = str.maketrans({
        '"': r'\"',
        '\\': r'\\',
    })
    for line in data:
        total += len(line)
        line = '"' + line.translate(map_table) + '"'
        cnt += len(line)
    print(cnt - total)
```

## Day 9: 

### Part 1:  Traveling Salesman Problem, dynamic programming, min heap, all pairs shortest distance, shortest route

```py
edge_encoded = parse.compile("{} to {} = {:d}")
with open('big.txt', 'r') as f:
    data = f.read().splitlines()
    # step 1: count number of distinct nodes
    seen = set()
    index = {}
    for line in data:
        u, v, w = edge_encoded.parse(line).fixed
        if u not in seen:
            index[u] = len(index)
            seen.add(u)
        if v not in seen:
            index[v] = len(index)
            seen.add(v)
    n = len(seen)
    # step 2: read in the all pairs shortest distance
    dist = [[0] * n for _ in range(n)]
    for line in data:
        u, v, w = edge_encoded.parse(line).fixed
        dist[index[u]][index[v]] = w
        dist[index[v]][index[u]] = w
    # step 3: run dynamic programming
    end_mask = (1 << n) - 1
    # start a trip from each node
    dp = [[math.inf] * n for _ in range(1 << n)]
    for src in range(n):
        dp[1 << src][src] = 0
    for mask in range(1 << src, 1 << n):
        for u in range(n):
            if dp[mask][u] == math.inf: continue
            for v in range(n):
                if (mask >> v) & 1: continue # already visited
                nmask = mask | (1 << v)
                dp[nmask][v] = min(dp[nmask][v], dp[mask][u] + dist[u][v])
    # for all routes that visit every node and the last visited node is i
    res = min(dp[end_mask])
    print(res)
```

### Solution 2:  Brute force every permutation

For part 2 you want to maximize the distance. try every permutation of nodes.  O(n!)

```py
edge_encoded = parse.compile("{} to {} = {:d}")
def path_length(dist, nodes):
    n = len(nodes)
    return sum(dist[nodes[i - 1]][nodes[i]] for i in range(1, n))
with open('big.txt', 'r') as f:
    data = f.read().splitlines()
    # step 1: count number of distinct nodes
    seen = set()
    index = {}
    for line in data:
        u, v, w = edge_encoded.parse(line).fixed
        if u not in seen:
            index[u] = len(index)
            seen.add(u)
        if v not in seen:
            index[v] = len(index)
            seen.add(v)
    n = len(seen)
    # step 2: read in the all pairs shortest distance
    dist = [[0] * n for _ in range(n)]
    for line in data:
        u, v, w = edge_encoded.parse(line).fixed
        dist[index[u]][index[v]] = w
        dist[index[v]][index[u]] = w
    # step 3: run dynamic programming
    end_mask = (1 << n) - 1
    # start a trip from each node
    p2 = -math.inf
    # try every permutation
    for perm in permutations(range(n)):
        p2 = max(p2, path_length(dist, perm))
    print("part 2:", p2)
```

## Day 10: 

### Part 2:  groupby, string with join

```py
with open("input.txt", "r") as f:
    num = f.read()
    def process(num, n):
        for _ in range(n):
            num = "".join([str(len(list(grp))) + k for k, grp in groupby(num)])
        return len(num)
    print(process(num, 40))
```

## Day 11: 

### Part 2, string, characters, modular, wrap around

```py
bad_letters = "ilo"

with open("input.txt", "r") as f:
    data = list(f.read())
    n = len(data)
    unicode = lambda ch: ord(ch) - ord('a')
    char = lambda val: chr(val + ord("a"))
    def valid(password):
        prev = "#"
        str8 = 0
        found_straight = False
        pairs = set()
        for ch in password:
            if ch in bad_letters: return False
            if unicode(ch) - unicode(prev) == 1: str8 += 1
            else: str8 = 1
            if str8 >= 3: found_straight = True
            if ch == prev: pairs.add(ch)
            prev = ch
        return found_straight and len(pairs) >= 2
    def increment():
        pivot = next(dropwhile(lambda i: data[i] == "z", reversed(range(n))))
        for i in range(pivot, n):
            ch = char((unicode(data[i]) + 1) % 26)
            data[i] = ch    
    while not valid(data):
        increment()
    increment()
    while not valid(data):
        increment()
    print("".join(data))
```

## Day 12: 

### Solution 1:  Use ast to read in a dictionary and lists, for json, use recursion to sum up all the children, and include if found red in a dictionary for the second part

```py
def recurse(object):
    res = 0
    if isinstance(object, int): return object
    elif isinstance(object, dict):
        for _, v in object.items():
            res += recurse(v)
    elif isinstance(object, list):
        for v in object: res += recurse(v)
    return res
def dfs(object):
    res = 0
    if isinstance(object, str) and object == "red": return 0, True
    if isinstance(object, int): return object, False
    if isinstance(object, dict):
        for v in object.values():
            value, fred = dfs(v)
            if fred: return 0, False
            res += value
    elif isinstance(object, list):
        for v in object: 
            value, fred = dfs(v)
            res += value
    return res, False
with open("big.txt", "r") as f:
    data = f.read()
    parsed_data = ast.literal_eval(data)
    part_1 = recurse(parsed_data)
    part_2, _ = dfs(parsed_data)
    print("part 1:", part_1)
    print("part 2:", part_2)
```

## Day 13: 

### Solution 1:  permutations, brute force, custom max function, circular array

```py
gain = compile("{} would gain {:d} happiness units by sitting next to {}.")
lose = compile("{} would lose {:d} happiness units by sitting next to {}.")
def solve(n, adj_mat):
    score = lambda arr: sum(adj_mat[arr[i]][arr[(i + 1) % n]] + adj_mat[arr[i]][arr[(i - 1) % n]] for i in range(n))
    return max(score(perm) for perm in permutations(range(n)))
def main():
    with open('big.txt', 'r') as f:
        data = f.read().splitlines()
        nodes = set()
        for line in data:
            g, l = gain.parse(line), lose.parse(line)
            if g:
                u, w, v = g.fixed
            else:
                u, w, v = l.fixed
            nodes.update((u, v))
        n = len(nodes)
        adj_mat = [[0] * (n + 1) for _ in range(n + 1)]
        nodes = list(nodes)
        index = {node: i for i, node in enumerate(nodes)}
        for line in data:
            g, l = gain.parse(line), lose.parse(line)
            if g:
                u, w, v = g.fixed
            else:
                u, w, v = l.fixed
                w = -w
            adj_mat[index[u]][index[v]] = w
        part_1 = solve(n, adj_mat)
        print("part 1:", part_1)
        part_2 = solve(n + 1, adj_mat)
        print("part 2:", part_2)
main()
```

## Day 14: 

### Solution 1: 

```py
reindeer = compile("{} can fly {:d} km/s for {:d} seconds, but then must rest for {:d} seconds.")
def solve(time, racers):
    ans = 0
    points = [0] * len(racers)
    for t in range(1, time + 1):
        leading_dist = 0
        leaders = []
        for i, (spd, dur, rest) in enumerate(racers):
            cycle = dur + rest
            cycles = t // cycle
            rem = t % cycle
            dist = cycles * spd * dur + spd * min(rem, dur)
            if dist > leading_dist:
                leading_dist = dist
                leaders.clear()
            if dist == leading_dist:
                leaders.append(i)
        ans = max(ans, leading_dist)
        for idx in leaders:
            points[idx] += 1
    return ans, max(points)
def main():
    with open('big.txt', 'r') as f:
        data = f.read().splitlines()
        racers = []
        for line in data:
            _, spd, dur, rest = reindeer.parse(line).fixed
            racers.append((spd, dur, rest))
        part_1, part_2 = solve(2_503, racers)
        print("part 1:", part_1)
        print("part 2:", part_2)
main()
```

## Day 15: 

### Solution 1:  memoization of bags, bag of the amount of teaspoons for each ingredient, iterate through every possible bag to find max, only 10^5 possibilities with 4 ingredients

```py
ingredient = compile("{}: capacity {:d}, durability {:d}, flavor {:d}, texture {:d}, calories {:d}")
with open('big.txt', 'r') as f:
    data = f.read().splitlines()
    ingreds = []
    for line in data:
        name, cap, dur, flav, tex, cal = ingredient.parse(line).fixed
        ingreds.append((cap, dur, flav, tex, cal))
    tot = 100
    n = len(ingreds)
    def score(items, part):
        sums = [0] * 5
        for i in range(5):
            for id, amt in enumerate(items):
                sums[i] += ingreds[id][i] * amt
        if part == 1: return math.prod([max(0, s) for s in sums[:-1]])
        return math.prod([max(0, s) for s in sums[:-1]]) if sums[-1] == 500 else 0
    init_bag = tuple([tot] + [0] * (n - 1))
    memo = {init_bag}
    stack = [init_bag]
    while stack:
        bag = list(stack.pop())
        if bag[0] == 0: continue
        bag[0] -= 1
        for i in range(1, n):
            bag[i] += 1
            nbag = tuple(bag)
            if nbag not in memo:
                memo.add(nbag)
                stack.append(nbag)
            bag[i] -= 1
    part_1 = part_2 = 0
    for bag in memo:
        part_1 = max(part_1, score(bag, 1))
        part_2 = max(part_2, score(bag, 2))
    print("part 1:", part_1)
    print("part 2:", part_2)
```

## Day 16: 

### Solution 1:  hashmap, check partial match

```py

with open("big.txt", "r") as f:
    data = f.read().splitlines()
    gifts = []
    for i, line in enumerate(data):
        j = line.index(":") + 2
        items = line[j:].split(", ")
        gift = {}
        for item in items:
            name, amt = item.split(": ")
            amt = int(amt)
            gift[name] = amt
        gifts.append(gift)
    target = {"children": 3, "cats": 7, "samoyeds": 2, "pomeranians": 3, "akitas": 0, "vizslas": 0, "goldfish": 5, "trees": 3, "cars": 2, "perfumes": 1}
    for i, partial_gift in enumerate(gifts, start = 1):
        if (
            all(target[key] < partial_gift.get(key, math.inf) for key in ["cats", "trees"]) and
            all(target[key] > partial_gift.get(key, -math.inf) for key in ["pomeranians", "goldfish"]) and
            all(target[key] == partial_gift.get(key, target[key]) for key in ["children", "samoyeds", "akitas", "vizslas", "cars", "perfumes"])
            ): print("part 2:", i)
        if all(target[key] == partial_gift[key] for key in partial_gift): print("part 1:", i)
```

## Day 17: 

### Solution 1:  dynamic programming with bags and counter

5.37 ms

```py
with open("big.txt", "r") as f:
    data = list(map(int, f.read().splitlines()))
    # O(N * M)
    N = 150
    M = len(data)
    dp = Counter({(0, 0): 1})
    for amt in data:
        ndp = dp.copy()
        for (cap, cnt), val in dp.items():
            if cap + amt > N: continue
            ndp[(cap + amt, cnt + 1)] += val
        dp = ndp
    part_1 = sum([val for (cap, cnt), val in dp.items() if cap == N])
    print("part 1:", part_1)
    key = min([(cap, cnt) for cap, cnt in dp.keys() if cap == N], key = lambda x: x[1])
    print("part 2:", dp[key])
```

### Solution 2: dynamic programming with table

13.8 ms

```py
with open("big.txt", "r") as f:
    data = list(map(int, f.read().splitlines()))
    # O(N * M)
    N = 150
    M = len(data)
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 1
    for amt in data:
        for cap in range(N, amt - 1, -1):
            for cnt in range(1, M + 1):
                dp[cap][cnt] += dp[cap - amt][cnt - 1]
    part_1 = sum(dp[N])
    print("part 1:", part_1)
    part_2 = next(val for val in dp[N] if val > 0)
    print("part 2:", part_2)
```

## Day 18: 

### Part 1: 

```cpp
const int STEPS = 100;
vector<vector<char>> grid;
int N;
const char ON = '#', OFF = '.';

void print(const vector<vector<char>>& g) {
    for (const auto& row : g) {
        for (char c : row) {
            cout << c;
        }
        cout << endl;
    }
}

bool inBounds(int r, int c) {
    return r >= 0 && r < N && c >= 0 && c < N;
}

vector<vector<char>> simulate() {
    vector<vector<char>> ans(N, vector<char>(N, '.'));
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int occupied = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int nr = r + dr, nc = c + dc;
                    if (!inBounds(nr, nc)) continue;
                    if (grid[nr][nc] == ON) occupied++;
                }
            }
            if (occupied == 3) ans[r][c] = ON;
            else if (occupied == 2) ans[r][c] = grid[r][c];
            else ans[r][c] = OFF;
        }
    }
    return ans;
}

int countOn() {
    int ans = 0;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (grid[r][c] == ON) {
                ans++;
            }
        }
    }
    return ans;
}

void solve() {
    string line;
    while (getline(cin, line)) {
        vector<char> row;
        for (char c : line) {
            row.emplace_back(c);
        }
        grid.emplace_back(row);
    }
    N = grid.size();
    for (int i = 0; i < STEPS; ++i) {
        vector<vector<char>> newGrid = simulate();
        swap(newGrid, grid);
    }
    int ans = countOn();
    debug(ans, "\n");
}
```

### Part 2: 

```cpp
const int STEPS = 100;
vector<vector<char>> grid;
int N;
const char ON = '#', OFF = '.';

bool inBounds(int r, int c) {
    return r >= 0 && r < N && c >= 0 && c < N;
}

bool isCorner(int r, int c) {
    return (r == 0 && c == 0) || (r == 0 && c == N - 1) || (r == N - 1 && c == 0) || (r == N - 1 && c == N - 1);
}

vector<vector<char>> simulate() {
    vector<vector<char>> ans(N, vector<char>(N, '.'));
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (isCorner(r, c)) {
                ans[r][c] = ON;
                continue;
            }
            int occupied = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int nr = r + dr, nc = c + dc;
                    if (!inBounds(nr, nc)) continue;
                    if (grid[nr][nc] == ON) occupied++;
                }
            }
            if (occupied == 3) ans[r][c] = ON;
            else if (occupied == 2) ans[r][c] = grid[r][c];
            else ans[r][c] = OFF;
        }
    }
    return ans;
}

int countOn() {
    int ans = 0;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (grid[r][c] == ON) {
                ans++;
            }
        }
    }
    return ans;
}

void solve() {
    string line;
    while (getline(cin, line)) {
        vector<char> row;
        for (char c : line) {
            row.emplace_back(c);
        }
        grid.emplace_back(row);
    }
    N = grid.size();
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (isCorner(r, c)) {
                grid[r][c] = ON;
            }
        }
    }
    for (int i = 0; i < STEPS; ++i) {
        vector<vector<char>> newGrid = simulate();
        swap(newGrid, grid);
    }
    int ans = countOn();
    debug(ans, "\n");
}
```

## Day 19: 

### Solution 1: 

```py

```

## Day 20: 

### Solution 1: 

```py

```

## Day 21: 

### Solution 1:  math ceiling, dictionary, combinations

```py
class Equipment:
    def __init__(self, name, cost, damage, armor):
        self.name, self.cost, self.damage, self.armor = name, cost, damage, armor
    def __repr__(self):
        return f"name = {self.name}, cost = {self.cost}, damage = {self.damage}, armor = {self.armor}"

with open("big.txt", "r") as f:
    data = f.read().splitlines()
    weapons = [Equipment(n, c, d, 0) for n, c, d in [("Dagger", 8, 4), ("Shortsword", 10, 5), ("Warhammer", 25, 6), ("Longsword", 40, 7), ("Greataxe", 74, 8)]]
    armor = [Equipment(n, c, 0, a) for n, c, a in [("Nothing", 0, 0), ("Leather", 13, 1), ("Chainmail", 31, 2), ("Splintmail", 53, 3), ("Bandedmail", 75, 4), ("Platemail", 102, 5)]]
    rings = [Equipment(n, c, d, a) for n, c, d, a in [("dmg+1", 25, 1, 0), ("dmg+2", 50, 2, 0), ("dmg+3", 100, 3, 0), ("armor+1", 20, 0, 1), ("armor+2", 40, 0, 2), ("armor+3", 80, 0, 3)]]
    enemy = {}
    for line in data:
        _, val = line.split(": ")
        if "Hit Points" in line:
            enemy["hp"] = int(val)
        elif "Damage" in line:
            enemy["dmg"] = int(val)
        else:
            enemy["armor"] = int(val)
    p1 = math.inf
    p2 = 0
    player = {"hp": 100, "dmg": 0, "armor": 0}
    for weapon, arm in product(weapons, armor):
        player["dmg"] = 0
        player["armor"] = 0
        player["dmg"] += weapon.damage
        player["armor"] += arm.armor
        for i in range(3):
            for ring in combinations(rings, i):
                player["dmg"] += sum(r.damage for r in ring)
                player["armor"] += sum(r.armor for r in ring)
                cost = weapon.cost + arm.cost + sum(r.cost for r in ring)
                pwinner = math.ceil(enemy["hp"] / max(1, player["dmg"] - enemy["armor"]))
                ewinner = math.ceil(player["hp"] / max(1, enemy["dmg"] - player["armor"]))
                if pwinner <= ewinner:
                    p1 = min(p1, cost)
                else:
                    p2 = max(p2, cost)
                player["dmg"] -= sum(r.damage for r in ring)
                player["armor"] -= sum(r.armor for r in ring)
    print("part 1:", p1)
    print("part 2:", p2)
    print(enemy)
```

## Day 22: 

### Solution 1:  dijkstra algorithm

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    enemy = {}
    for line in data:
        attribute, val = line.split(": ")
        enemy[attribute] = int(val)
    print(enemy)
    # (consumed mana, your hp, enemy hp, mana, shield, poison, recharge)
    init_state = (0, 50, enemy["Hit Points"], 500, 0, 0, 0)
    min_heap = [init_state]
    cnt = 0
    vis = set()
    while min_heap:
        cost, hp, rem_hp, mana, shield, poison, recharge = heapq.heappop(min_heap)
        if (cost, hp, rem_hp, mana, shield, poison, recharge) in vis: continue
        vis.add((cost, hp, rem_hp, mana, shield, poison, recharge))
        cnt += 1
        if cnt % 100_000 == 0: print(cnt, cost)
        if mana < 0: continue
        if rem_hp <= 0: break
        if hp <= 0: continue
        defense = 7 if shield > 1 else 0
        enemy_dmg = max(1, enemy["Damage"] - defense)
        mana += min(202, recharge * 101)
        rem_hp -= min(6, poison * 3)
        # missile
        heapq.heappush(min_heap, (cost + 53, hp - enemy_dmg, rem_hp - 4, mana - 53, max(0, shield - 2), max(0, poison - 2), max(0, recharge - 2)))
        # drain
        heapq.heappush(min_heap, (cost + 73, hp - enemy_dmg + 2, rem_hp - 2, mana - 73, max(0, shield - 2), max(0, poison - 2), max(0, recharge - 2)))
        # shield
        if shield == 0:
            heapq.heappush(min_heap, (cost + 113, hp - max(1, enemy["Damage"] - 7), rem_hp, mana - 113, 5, max(0, poison - 2), max(0, recharge - 2)))
        # poison
        if poison == 0:
            heapq.heappush(min_heap, (cost + 173, hp - enemy_dmg, rem_hp - 3, mana - 173, max(0, shield - 2), 5, max(0, recharge - 2)))
        # recharge
        if mana >= 229 and recharge == 0:
            heapq.heappush(min_heap, (cost + 229, hp - enemy_dmg, rem_hp, mana - 229 + 101, max(0, shield - 2), max(0, poison - 2), 4))
    print("part 1:", cost)
```

```py
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    enemy = {}
    for line in data:
        attribute, val = line.split(": ")
        enemy[attribute] = int(val)
    print(enemy)
    # (consumed mana, your hp, enemy hp, mana, shield, poison, recharge)
    init_state = (0, 50, enemy["Hit Points"], 500, 0, 0, 0)
    min_heap = [init_state]
    cnt = 0
    vis = set()
    while min_heap:
        cost, hp, rem_hp, mana, shield, poison, recharge = heapq.heappop(min_heap)
        if (cost, hp, rem_hp, mana, shield, poison, recharge) in vis: continue
        vis.add((cost, hp, rem_hp, mana, shield, poison, recharge))
        cnt += 1
        if cnt % 100_000 == 0: print(cnt, cost)
        hp -= 1
        if mana < 0: continue
        if rem_hp <= 0: 
            print("cost", cost, "hp", hp, "rem_hp", rem_hp, "mana", mana, "shield, poison, recharge", shield, poison, recharge)
            break
        if hp <= 0: continue
        defense = 7 if shield > 1 else 0
        enemy_dmg = max(1, enemy["Damage"] - defense)
        mana += min(202, recharge * 101)
        rem_hp -= min(6, poison * 3)
        # missile
        heapq.heappush(min_heap, (cost + 53, hp - enemy_dmg, rem_hp - 4, mana - 53, max(0, shield - 2), max(0, poison - 2), max(0, recharge - 2)))
        # drain
        heapq.heappush(min_heap, (cost + 73, hp - enemy_dmg + 2, rem_hp - 2, mana - 73, max(0, shield - 2), max(0, poison - 2), max(0, recharge - 2)))
        # shield
        if shield == 0:
            heapq.heappush(min_heap, (cost + 113, hp - max(1, enemy["Damage"] - 7), rem_hp, mana - 113, 5, max(0, poison - 2), max(0, recharge - 2)))
        # poison
        if poison == 0:
            heapq.heappush(min_heap, (cost + 173, hp - enemy_dmg, rem_hp - 3, mana - 173, max(0, shield - 2), 5, max(0, recharge - 2)))
        # recharge
        if mana >= 229 and recharge == 0:
            heapq.heappush(min_heap, (cost + 229, hp - enemy_dmg, rem_hp, mana - 229 + 101, max(0, shield - 2), max(0, poison - 2), 4))
    print("part 2:", cost)
```

## Day 23: 

### Solution 1:  dataclass, python switch statements, simulation

```py
jio = parse.compile("jio {}, {:d}")
jmp = parse.compile("jmp {:d}")
tpl = parse.compile("tpl {}")
inc = parse.compile("inc {}")
hlf = parse.compile("hlf {}")
jie = parse.compile("jie {}, {:d}")
@dataclass
class Instruction:
    name: str
    reg: str = field(default = None)
    offset: int = field(default = 1)
with open("big.txt", "r") as f:
    data = f.read().splitlines()
    instructions = []
    registers = Counter()
    for line in data:
        if jmp.parse(line):
            offset = jmp.parse(line).fixed[0]
            instructions.append(Instruction(name = "jmp", offset = offset))
        elif jio.parse(line):
            reg, offset = jio.parse(line).fixed
            instructions.append(Instruction(name = "jio", reg = reg, offset = offset))
            registers[reg] = 0
        elif tpl.parse(line):
            reg = tpl.parse(line).fixed[0]
            instructions.append(Instruction(name = "tpl", reg = reg))
            registers[reg] = 0
        elif inc.parse(line):
            reg = inc.parse(line).fixed[0]
            instructions.append(Instruction(name = "inc", reg = reg))
            registers[reg] = 0
        elif hlf.parse(line):
            reg = hlf.parse(line).fixed[0]
            instructions.append(Instruction(name = "hlf", reg = reg))
            registers[reg] = 0
        else:
            reg, offset = jie.parse(line).fixed
            instructions.append(Instruction(name = "jie", reg = reg, offset = offset))
            registers[reg] = 0
    n = len(instructions)
    registers["a"] = 1 # for part 2
    idx = cnt = 0
    while idx < n:
        cnt += 1
        match instructions[idx].name:
            case "jmp": idx += instructions[idx].offset
            case "jio": idx += instructions[idx].offset if registers[instructions[idx].reg] == 1 else 1
            case "tpl": registers[instructions[idx].reg] *= 3; idx += 1
            case "inc": registers[instructions[idx].reg] += 1; idx += 1
            case "hlf": registers[instructions[idx].reg] >>= 1; idx += 1
            case "jie": idx += instructions[idx].offset if registers[instructions[idx].reg] % 2 == 0 else 1
    print("number of iterations:", cnt)
    print("part 1:", registers["b"])
```

## Day 24: 

### Solution 1: 

```py

```

## Day 25: 

### Solution 1:  brute force

```py
def solve():
    m = 252533
    d = 33554393
    val = 20151125
    er, ec = 2977, 3082
    for sum_ in range(1, er + ec + 1):
        for i in range(sum_ + 1):
            r, c = sum_ - i, i
            val = (val * m) % d
            if (r, c) == (er, ec): return val
solve()
```