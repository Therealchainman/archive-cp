# Leetcode Problems Solved in 2026

## 961. N-Repeated Element in Size 2N Array

### Solution 1: Sliding Window with Frequency Map

```cpp
class Solution {
public:
    int repeatedNTimes(vector<int>& nums) {
        int N = nums.size();
        unordered_map<int, int> freq;
        for (int i = 0; i < N; ++i) {
            freq[nums[i]]++;
            if (freq[nums[i]] > 1) return nums[i];
            if (i >= 3) freq[nums[i - 3]]--;
        }
        return -1;
    }
};
```

## 1390. Four Divisors

### Solution 1: number theory, divisors

```cpp
class Solution {
public:
    int sumFourDivisors(vector<int>& nums) {
        int ans = 0;
        for (int x : nums) {
            int last = 0;
            for (int d = 2; d * d <= x; ++d) {
                if (x % d == 0) {
                    if (last) {
                        last = 0;
                        break;
                    }
                    last = d;
                }
            }
            if (last > 0 && last != x / last) {
                ans += 1 + last + x / last + x;
            }
        }
        return ans;
    }
};
```

## 1411. Number of Ways to Paint N × 3 Grid

### Solution 1: dynamic programming, counting, base 3 mask to represent colorings

```cpp
const int MOD = 1e9 + 7;
class Solution {
private:
    bool isCompatible(int x, int y) {
        for (int i = 0; i < 3; ++i) {
            if (x % 3 == y % 3) return false;
            x /= 3;
            y /= 3;
        }
        return true;
    }
public:
    int numOfWays(int n) {
        vector<int> positions;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (i == j) continue;
                for (int k = 0; k < 3; ++k) {
                    if (k == j) continue;
                    int mask = 9 * i + 3 * j + k;
                    positions.emplace_back(mask);
                }
            }
        }
        int N = positions.size();
        vector<int> dp(N, 1), ndp(N, 0);
        while (--n) {
            ndp.assign(N, 0);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (!isCompatible(positions[i], positions[j])) continue;
                    ndp[j] = (ndp[j] + dp[i]) % MOD;
                }
            }
            swap(ndp, dp);
        }
        int ans = accumulate(dp.begin(), dp.end(), 0, [](int total, int x) { return (total + x) % MOD; });
        return ans;
    }
};
```

## 1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold

### Solution 1: row and column prefix sums, three nested loops

```cpp
class Solution {
public:
    int maxSideLength(vector<vector<int>>& mat, int threshold) {
        int N = mat.size(), M = mat[0].size();
        vector<vector<int>> colSums(N + 1, vector<int>(M, 0)), rowSums(N, vector<int>(M + 1, 0));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                colSums[i + 1][j] += colSums[i][j] + mat[i][j];
                rowSums[i][j + 1] += rowSums[i][j] + mat[i][j];
            }
        }
        int ans = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                int cand = 0;
                for (int k = 0; i + k < N && j + k < M; ++k) {
                    int rsum = rowSums[i + k][j + k + 1] - rowSums[i + k][j];
                    int csum = colSums[i + k + 1][j + k] - colSums[i][j + k];
                    cand += rsum + csum - mat[i + k][j + k];
                    if (cand > threshold) break;
                    ans = max(ans, k + 1);
                }
            }
        }
        return ans;
    }
};
```

### Solution 2: 2d prefix sums

Side length can only get better, so no reason to try values less than ans.

```cpp
class Solution {
private:
    int query(const vector<vector<int>> &ps, int r1, int c1, int r2, int c2) {
        if (r1 > r2 || c1 > c2) return 0;
        return ps[r2 + 1][c2 + 1] - ps[r1][c2 + 1] - ps[r2 + 1][c1] + ps[r1][c1];
    }
public:
    int maxSideLength(vector<vector<int>>& mat, int threshold) {
        int R = mat.size(), C = mat[0].size();
        vector<vector<int>> ps(R + 1, vector<int>(C + 1, 0));
        for (int r = 1; r <= R; r++) {
            for (int c = 1; c <= C; c++) {
                ps[r][c] = ps[r - 1][c] + ps[r][c - 1] - ps[r - 1][c - 1] + mat[r - 1][c - 1];
            }
        }
        int ans = 0;
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) {
                int cand = 0;
                for (int k = ans; i + k < R && j + k < C; ++k) {
                    int cand = query(ps, i, j, i + k, j + k);
                    if (cand > threshold) break;
                    ans = max(ans, k + 1);
                }
            }
        }
        return ans;
    }
};
```

## 2977. Minimum Cost to Convert String II

### Solution 1: polynomial rolling hash, weighted directed graph, dijkstra's algorithm, all pairs shortest path, dynamic programming

```cpp
using int64 = long long;
const int64 INF = numeric_limits<int64>::max();
const int64 p = 31, MOD1 = pow(2, 43) - 1;
class Solution {
private:
    unordered_map<int64, int> nodes;
    vector<vector<pair<int, int>>> adj;
    int V;
    vector<vector<int64>> dist;
    int coefficient(char ch) {
        return ch - 'a' + 1;
    }
    
    void dijkstra(int src) {
        priority_queue<pair<int64, int>, vector<pair<int64, int>>, greater<pair<int64, int>>> minheap;
        minheap.emplace(0, src);
        dist[src][src] = 0;
        while (!minheap.empty()) {
            auto [c, u] = minheap.top();
            minheap.pop();
            if (c > dist[src][u]) continue;
            for (auto [v, w] : adj[u]) {
                if (c + w < dist[src][v]) {
                    dist[src][v] = c + w;
                    minheap.emplace(c + w, v);
                }
            }
        }
    }
    int64 stringHash(const string& s) {
        int N = s.size();
        int64 hash = 0;
        for (int i = N - 1; i >= 0; --i) {
            hash = (p * hash + coefficient(s[i])) % MOD1;
        }
        return hash;
    }
public:
    int64 minimumCost(string source, string target, vector<string>& original, vector<string>& changed, vector<int>& cost) {
        int N = source.size();
        vector<int64> POW(N);
        POW[0] = 1;
        for (int i = 1; i < N; i++) {
            POW[i] = (POW[i - 1] * p) % MOD1;
        }
        for (int i = 0; i < original.size(); ++i) {
            int64 hu = stringHash(original[i]), hv = stringHash(changed[i]);
            if (nodes.find(hu) == nodes.end()) {
                adj.emplace_back(vector<pair<int, int>>());
                nodes[hu] = nodes.size();
            }
            if (nodes.find(hv) == nodes.end()) {
                adj.emplace_back(vector<pair<int, int>>());
                nodes[hv] = nodes.size();
            }
            int u = nodes[hu], v = nodes[hv], w = cost[i];
            adj[u].emplace_back(v, w);
        }
        V = nodes.size();
        dist.assign(V, vector<int64>(V, INF));
        for (int i = 0; i < V; ++i) {
            dijkstra(i);
        }
        vector<int64> dp(N + 1, INF);
        dp[0] = 0;
        for (int i = 1; i <= N; ++i) {
            int64 hs = 0, ht = 0;
            for (int j = i; j > 0; --j) {
                hs = (p * hs + coefficient(source[j - 1])) % MOD1;
                ht = (p * ht + coefficient(target[j - 1])) % MOD1;
                if (dp[j - 1] == INF) continue;
                if (hs == ht) {
                    dp[i] = min(dp[i], dp[j - 1]);
                } else if (nodes.find(hs) != nodes.end() && nodes.find(ht) != nodes.end()) {
                    int u = nodes[hs], v = nodes[ht];
                    if (dist[u][v] != INF) dp[i] = min(dp[i], dp[j - 1] + dist[u][v]);
                }
            }
        }
        return dp[N] < INF ? dp[N] : -1;
    }
};
```

## 2693. Call Function with Custom Context

### Solution 1: this.call, this context

spread operator to pass multiple args into this.call, and applying some context into the calling of the function. 

```ts
type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };

interface Function {
    callPolyfill(context: Record<string, JSONValue>, ...args: JSONValue[]): JSONValue;
}


Function.prototype.callPolyfill = function(context, ...args): JSONValue {
    return this.call(context, ...args);
}

```

### Solution 2: Symbol to avoid property name collision

```ts
type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };

interface Function {
    callPolyfill(context: Record<string, JSONValue>, ...args: JSONValue[]): JSONValue;
}


Function.prototype.callPolyfill = function(context, ...args): JSONValue {
    const ctx = context as Record<PropertyKey, any>;
    const s = Symbol();
    ctx[s] = this;
    try {
        return ctx[s](...args);
    } finally {
        delete ctx[s];
    }
}
```

## 2777. Date Range Generator

### Solution 1: generator, setDate, toISOString, Date object

```ts
function* dateRangeGenerator(start: string, end: string, step: number) : Generator<string> {
    const startDate = new Date(start), endDate = new Date(end);
    while (startDate <= endDate) {
        yield startDate.toISOString().split('T')[0].trim();
        startDate.setDate(startDate.getDate() + step);
    }
};
```

## 2795. Parallel Execution of Promises for Individual Results Retrieval

### Solution 1: promises, Promise.resolve, finally, forEach with side effects

You use the forEach because you aren't returning anything from it, just using it to iterate over the functions array and trigger the promises. The results array is populated in place based on the index, so the order of results matches the order of the input functions.

This is a way to ensure that you can run all the promises in parallel and collect their results (or errors) while maintaining the order of the original functions.

```ts
type FulfilledObj = {
    status: 'fulfilled';
    value: string;
}
type RejectedObj = {
    status: 'rejected';
    reason: string;
}
type Obj = FulfilledObj | RejectedObj;

function promiseAllSettled(functions: Function[]): Promise<Obj[]> {
    const results: Obj[] = [];
    let cnt = 0;
    return new Promise(resolve => {
        functions.forEach((fn, i) => {
            Promise.resolve(fn())
            .then(data => {
                results[i] = {
                    status: 'fulfilled',
                    value: data
                }
            })
            .catch(err => {
                results[i] = {
                    status: 'rejected',
                    reason: err
                }
            })
            .finally(() => {
                if (++cnt === functions.length) {
                    resolve(results);
                }
            });
        });
    });
};
```

## 2823. Deep Object Filter

### Solution 1: recursion, Object.entries, Array.isArray, undefined, null

the type of null is object

```ts
type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };
type Obj = Record<string, JSONValue> | Array<JSONValue>

function deepFilter(obj: Obj, fn: Function): Obj | undefined {
    if (obj === null || (typeof obj !== 'object' && !Array.isArray(obj))) return fn(obj) ? obj : undefined;
    if (Array.isArray(obj)) {
        const arr = []
        for (const x of obj) {
            const v = deepFilter(x as Obj, fn);
            if (v === undefined) continue;
            if (Array.isArray(v) && v.length === 0) continue;
            arr.push(v);
        }
        return arr.length > 0 ? arr : undefined;
    }
    const res = {}
    for (const [key, val] of Object.entries(obj)) {
        const v = deepFilter(val as Obj, fn);
        if (v === undefined) continue;
        if (typeof v === "object" && !Array.isArray(v) && v !== null && Object.keys(v).length === 0) continue;
        res[key] = v;
    }
    return Object.keys(res).length > 0 ? res : undefined;
};
```

## 2803. Factorial Generator

### Solution 1: generator

```ts
function* factorial(n: number): Generator<number> {
    if (n === 0) n++;
    let ans = 1;
    for (let x = 1; x <= n; ++x) {
        ans *= x;
        yield ans;
    }
};
```

## 2797. Partial Function with Placeholders

### Solution 1: closure, rest parameters, placeholder replacement

```ts
type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };
type Fn = (...args: JSONValue[]) => JSONValue

function partial(fn: Fn, args: JSONValue[]): Fn {
    return function(...restArgs) {
        let i = 0;
        for (let j = 0; j < args.length; ++j) {
            if (args[j] === '_') args[j] = restArgs[i++];
        }
        while (i < restArgs.length) {
            args.push(restArgs[i++]);
        }
        return fn(...args);
    }
};
```

## 2796. Repeat String

### Solution 1: recursion, String.prototype.valueOf

```ts
interface String {
    replicate(times: number): string;
}

String.prototype.replicate = function(times): string {
    return times > 0 ? this.valueOf() + this.replicate(times - 1) : "";
}
```

## 2794. Create Object from Two Arrays

### Solution 1: loop, javascript object

```ts
type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };

function createObject(keysArr: JSONValue[], valuesArr: JSONValue[]): Record<string, JSONValue> {
    const res = {};
    for (let i = 0; i < keysArr.length; ++i) {
        const key = String(keysArr[i]);
        if (res[key] !== undefined) continue;
        res[key] = valuesArr[i];
    }
    return res;
};
```

## 2822. Inversion of Object

### Solution 1: javascript object, Object.entries, typeof

```ts
type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };
type Obj = Record<string, JSONValue> | Array<JSONValue>

function invertObject(obj: Obj): Record<string, JSONValue> {
    const res = {}
    for (const [key, value] of Object.entries(obj)) {
        const val = String(value)
        const existing = res[val];
        if (existing === undefined) {
            res[val] = key;
        } else if (typeof existing === 'string') {
            res[val] = [existing, key];
        } else {
            res[val].push(key);
        }
    }
    return res;
};
```

## 2821. Delay the Resolution of Each Promise

### Solution 1: promises, setTimeout, map

```ts
type Fn = () => Promise<any>

function delayAll(functions: Fn[], ms: number): Fn[] {
    return functions.map(fn => () => new Promise((resolve, reject) => setTimeout(() => fn().then(resolve).catch(reject), ms)));
};
```

## 2775. Undefined to Null

### Solution 1: undefined, null, recursion, Object.entries, Array.isArray

in JSON.stringify undefined is omitted, while null is preserved.

```ts
type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };
type Value = undefined | null | boolean | number | string | Value[] | { [key: string]: Value };

type Obj1 = Record<string, Value> | Array<Value>
type Obj2 = Record<string, JSONValue> | Array<JSONValue>

function undefinedToNull(obj: Obj1): Obj2 {
    console.log(obj)
    if (obj === undefined) return null;
    if (obj === null || typeof obj !== 'object') return obj;
    if (Array.isArray(obj)) return obj.map(item => undefinedToNull(item as Obj1));
    const res = {};
    for (const [key, value] of Object.entries(obj)) {
        res[key] = undefinedToNull(value as Obj1);
    }
    return res;
};
```

## 2776. Convert Callback Based Function to Promise Based Function

### Solution 1: 

not sure yet.

```ts
type CallbackFn = (
    next: (data: number, error: string) => void, 
    ...args: number[]
) => void
type Promisified = (...args: number[]) => Promise<number>

function promisify(fn: CallbackFn): Promisified {
    console.log(fn);
    return async function(...args) {
        console.log(args, fn(2, ""));
        return new Promise(resolve => resolve(args.reduce((total, x) => total * x, 1)));
    };
};

```

## 2804. Array Prototype ForEach

### Solution 1: bind, for loop, callback

```ts
type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };
type Callback = (currentValue: JSONValue, index: number, array: JSONValue[]) => any
type Context = Record<string, JSONValue>

Array.prototype.forEach = function(callback: Callback, context: Context): void {
    const cb = callback.bind(context);
    const arr = this;
    for (let i = 0; i < arr.length; ++i) {
        cb(arr[i], i, arr);
    }
}
```

## 2805. Custom Interval

### Solution 1: map, setTimeout, recursion, dynamic delay

```ts
const customIntervalModule = (() => {
    let intervalCounter = 0;
    const timeoutMap = new Map<number, NodeJS.Timeout>();
    function customInterval(fn: Function, delay: number, period: number): number {
        const customId = intervalCounter++;
        let count = 0;
        const dfs = (() => {
            const timeoutId = setTimeout(() => {
                fn();
                dfs();
            }, delay + period * count);
            count++;
            timeoutMap.set(customId, timeoutId);
        })
        dfs();
        return customId;
    }

    function customClearInterval(id: number): void {
        clearTimeout(timeoutMap.get(id));
    }
    return {
        customInterval,
        customClearInterval
    }
})();

const customInterval = customIntervalModule.customInterval;
const customClearInterval = customIntervalModule.customClearInterval;
```

## 2758. Next Day

### Solution 1: Date object, next day, ISOString

Learned that setDate is in-place, and this is best way to increment by days.
toISOString() is a good way to get the string in YYYY-MM-DD format.

```ts
Date.prototype.nextDay = function() {
    this.setDate(this.getDate() + 1);
    return this.toISOString().slice(0, 10);
}
```

## 2774. Array Upper Bound

### Solution 1: binary search

```ts
interface Array<T> {
    upperBound(target: number): number;
}

function lowerBound(arr: number[], target: number) {
    const N = arr.length;
    let lo = 0, hi = N;
    while (lo < hi) {
        const mid = lo + Math.trunc((hi - lo) / 2);
        if (arr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

function upperBound(arr: number[], target: number) {
    const N = arr.length;
    let lo = 0, hi = N;
    while (lo < hi) {
        const mid = lo + Math.trunc((hi - lo) / 2);
        if (arr[mid] <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

Array.prototype.upperBound = function(target): number {
    const i = lowerBound(this, target), j = upperBound(this, target);
    return j > i ? j - 1 : -1;
};
```

## 2618. Check if Object Instance of Class

### Solution 1: contructor function, object, primitives, 

constructor function is provided, and want to determine if the object is an instance of that class.

primitives are not consider Object type, such as 5, "test", true, they are primitive data types.  And they are not going to be equal to Number or String or Boolean.

for primitives that are not Object you can convert to Object, with Object(obj)

```ts
function checkIfInstanceOf(obj: any, classFunction: any): boolean {
    if (typeof classFunction !== "function" || obj == null || classFunction == null) return false;
    return Object(obj) instanceof classFunction;
};
```

## 2759. Convert JSON String to Object

### Solution 1: 

```ts

```

## 2624. Snail Traversal

### Solution 1: 2d array, mapping 2d to 1d index

```ts
interface Array<T> {
    snail(rowsCount: number, colsCount: number): number[][];
}


Array.prototype.snail = function(rowsCount: number, colsCount: number): number[][] {
    const res: number[][] = Array.from({ length: rowsCount }, () => Array(colsCount));
    if (rowsCount * colsCount !== this.length) return [];
    for (let c = 0; c < colsCount; ++c) {
        for (let i = 0; i < rowsCount; ++i) {
            const r = c % 2 == 0 ? i : rowsCount - i - 1;
            res[r][c] = this[rowsCount * c + i];
        }
    }
    return res;
}
```

## 2628. JSON Deep Equal

### Solution 1: object, array, deeply equal, set

```ts
type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };

function areDeeplyEqual(o1: JSONValue, o2: JSONValue): boolean {
    if (typeof o1 !== "object" || typeof o2 !== "object" || o1 == null || o2 == null) {
        return o1 === o2;
    }
    if (Array.isArray(o1) || Array.isArray(o2)) {
        if (!Array.isArray(o1) || !Array.isArray(o2)) return false;
        if (o1.length != o2.length) return false;
        for (let i = 0; i < o1.length; ++i) {
            if (!areDeeplyEqual(o1[i], o2[i])) return false;
        }
    }
    const keys = new Set([...Object.keys(o1), ...Object.keys(o2)]);
    for (const k of keys) {
        if (!areDeeplyEqual(o1[k], o2[k])) return false;
    }
    return true;
};
```

## 761. Special Binary String

### Solution 1: recursion, string manipulation, balancing

```cpp
class Solution {
public:
    string makeLargestSpecial(string s) {
        int N = s.size();
        if (N == 2) return s;
        vector<string> specialArr;
        int bal = 0;
        for (int i = 0, j = 0; i < N; ++i) {
            if (s[i] == '1') bal++;
            else bal--;
            if (bal == 0) {`
                string inner = s.substr(j + 1, i - j - 1);
                specialArr.emplace_back('1' + makeLargestSpecial(inner) + '`0');
                j = i + 1;
            }
        }
        sort(specialArr.rbegin(), specialArr.rend());
        string ans = "";
        for (const string& s : specialArr) {
            ans += s;
        }
        return ans;
    }
};
```

## 762. Prime Number of Set Bits in Binary Representation

### Solution 1: bit manipulation, prime numbers

This just uses fact that 665772 is a bitmask with bits set at prime indices, so we can check if the number of set bits is prime by checking if the corresponding bit in 665772 is set.

```cpp
class Solution {
public:
    int countPrimeSetBits(int left, int right) {
        int ans = 0;
        while (left <= right) {
            ans += (665772 >> __builtin_popcount(left++)) & 1; 
        }
        return ans;
    }
};
```

## 868. Binary Gap

### Solution 1: bit manipulation

```cpp
class Solution {
public:
    int binaryGap(int n) {
        bool found = false;
        int ans = 0, cur = 0;
        while (n > 0) {
            if (n & 1) {
                if (found) ans = max(ans, cur);
                found = true;
                cur = 0;
            }
            cur++;
            n >>= 1;
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 173

## Q1. Reverse String Prefix

### Solution 1: string manipulation, reverse prefix

This is just split, reverse, and glue back together.

Take the first `k` characters, reverse that substring, then append the untouched suffix starting at `k`. Since the code never touches characters after the prefix, it matches the operation directly.

```cpp
class Solution {
public:
    string reversePrefix(string s, int k) {
        string t = s.substr(0, k);
        reverse(t.begin(), t.end());
        string ans = t + s.substr(k);
        return ans;
    }
};
```

## Q2. Minimum Subarray Length With Distinct Sum At Least K

### Solution 1: sliding window, frequency map, distinct-sum sum

The window keeps two pieces of information:

- `freq[x]`: how many times `x` appears in the current window
- `wsum`: sum of the distinct values currently present in the window

When `nums[r]` enters the window, it only increases `wsum` if it is the first copy. Then we greedily shrink from the left while either:

- `nums[l]` is duplicated, so removing it does not change the distinct-value sum
- or removing `nums[l]` still leaves the distinct sum at least `k`

After that loop, the window is the shortest valid window ending at `r`, so if `wsum >= k` we can update the answer with `r - l + 1`.

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int minLength(vector<int>& nums, int k) {
        int N = nums.size();
        int ans = INF, wsum = 0;
        unordered_map<int, int> freq;
        for (int l = 0, r = 0; r < N; ++r) {
            freq[nums[r]]++;
            if (freq[nums[r]] == 1) wsum += nums[r];
            while (freq[nums[l]] > 1 || wsum - nums[l] >= k) {
                freq[nums[l]]--;
                if (freq[nums[l]] == 0) wsum -= nums[l];
                l++;
            }
            if (wsum >= k) {
                ans = min(ans, r - l + 1);
            }
        }
        return ans < INF ? ans : -1;
    }
};
```

## Q3. Find Maximum Value in a Constrained Sequence

### Solution 1: two passes, propagate upper bounds from both sides

Each restriction says position `i` cannot exceed some value `x`, and adjacent positions can only change according to `diff`.

So for every index we compute the best upper bound coming from the left and from the right:

- `pref[i]`: tightest value allowed at `i` after propagating constraints left to right
- `suf[i]`: tightest value allowed at `i` after propagating constraints right to left

`pref[0] = 0` fixes the starting value. During the left-to-right pass, position `i + 1` cannot be more than `pref[i] + diff[i]`. The right-to-left pass does the symmetric update with `suf[i + 1] + diff[i]`.

In the final array, a position must satisfy both directions, so its maximum feasible value is `min(pref[i], suf[i])`. Taking the maximum of that over all indices gives the answer.

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int findMaxVal(int n, vector<vector<int>>& restrictions, vector<int>& diff) {
        vector<int> pref(n, INF), suf(n, INF);
        int ans = 0;
        pref[0] = 0;
        for (const auto &vec : restrictions) {
            int i = vec[0], x = vec[1];
            pref[i] = suf[i] = x;
        }
        for (int i = 0; i + 1 < n; ++i) {
            pref[i + 1] = min(pref[i + 1], pref[i] + diff[i]);
        }
        for (int i = n - 2; i >= 0; --i) {
            if (suf[i + 1] == INF) continue;
            suf[i] = min(suf[i], suf[i + 1] + diff[i]);
        }
        for (int i = 0; i < n; ++i) {
            ans = max(ans, min(pref[i], suf[i]));
        }
        return ans;
    }
};
```

## Q4. Count Routes to Climb a Rectangular Grid

### Solution 1: dynamic programming by rows, prefix sums for range transitions

This DP goes from the bottom row upward.

For each row, there are two kinds of transitions:

- come from the row below
- move to another hold in the same row

If we move between adjacent rows, the vertical distance is `1`, so the largest allowed horizontal shift is `floor(sqrt(d * d - 1))`. That is what `delta` stores. So the first pass on a row computes how many ways each open cell can be reached from the row below, using a prefix-sum array to query every valid column interval in `O(1)`.

After that, the second pass adds routes that make one extra move inside the same row. For that transition the full distance budget is horizontal, so the reachable interval is `[c - d, c + d]`. The code subtracts the single-cell range at `c` so we do not count staying on the same hold.

Both passes store row values as prefix sums, which makes each interval query constant time. Processing all rows this way gives an `O(RC)` solution.

```cpp
const int MOD = 1e9 + 7;
class Solution {
private:
    int rangeSum(const vector<int>& psum, int l, int r) {
        int ans = psum[r];
        if (l > 0) ans -= psum[l - 1];
        if (ans < 0) ans += MOD;
        return ans;
    }
public:
    int numberOfRoutes(vector<string>& grid, int d) {
        int R = grid.size(), C = grid[0].size();
        vector<int> dp(C, 0), ndp(C, 0);
        int delta = sqrt(d * d - 1);
        for (int r = R - 1; r >= 0; --r) {
            ndp.assign(C, 0);
            for (int c = 0; c < C; ++c) {
                if (grid[r][c] == '.') {
                    if (r == R - 1) {
                        ndp[c] = 1;
                    } else {
                        ndp[c] = rangeSum(dp, max(0, c - delta), min(C - 1, c + delta));
                    }
                    if (ndp[c] < 0) ndp[c] += MOD;
                }
                if (c > 0) ndp[c] += ndp[c - 1];
                ndp[c] %= MOD;
            }
            swap(ndp, dp);
            ndp.assign(C, 0);
            for (int c = 0; c < C; ++c) {
                if (grid[r][c] == '.') ndp[c] = rangeSum(dp, max(0, c - d), min(C - 1, c + d)) - rangeSum(dp, c, c);
                if (ndp[c] < 0) ndp[c] += MOD;
                if (c > 0) ndp[c] += ndp[c - 1];
                ndp[c] %= MOD;
            }
            for (int c = 0; c < C; ++c) {
                dp[c] += ndp[c];
                dp[c] %= MOD;
            }
        }
        return dp.back();
    }
};
```

# Leetcode Biweekly Contest 174

## Q1. Best Reachable Tower

### Solution 1: greedy scan, manhattan distance, lexicographic tie-break

We only need to inspect each tower once. A tower is a candidate exactly when its Manhattan distance from `center` is at most `radius`, and among those candidates we want the one with maximum quality.

So the scan keeps the best reachable tower seen so far. The `q < best` check skips towers that cannot improve the answer, and when `q == best` the vector comparison enforces the lexicographically smallest coordinate. If no tower passes the distance check, the initial `[-1, -1]` remains.

```cpp
class Solution {
private:
    int manhattanDistance(int x1, int y1, int x2, int y2) {
        return abs(x1 - x2) + abs(y1 - y2);
    }
public:
    vector<int> bestTower(vector<vector<int>>& towers, vector<int>& center, int radius) {
        int best = -1;
        vector<int> ans = {-1, -1};
        int cx = center[0], cy = center[1];
        for (const auto &tower : towers) {
            int x = tower[0], y = tower[1], q = tower[2]; 
            if (q < best) continue;
            vector<int> cand = {x, y};
            if (q == best && cand > ans) continue;
            if (manhattanDistance(x, y, cx, cy) <= radius) {
                best = q;
                ans = cand;
            }
        }
        return ans;
    }
};
```

## Q2. Minimum Operations to Reach Target Array

### Solution 1: hash set, distinct mismatching values

The key observation is that one operation is tied to a value from `nums`, not to an individual index. If some value `x` appears in several maximal segments, choosing `x` fixes all of those mismatching `x`-segments at once by replacing them with the corresponding values from `target`.

That means every value that appears at a mismatching position must be chosen at least once, and choosing it once is already enough. So the answer is exactly the number of distinct `nums[i]` values with `nums[i] != target[i]`, which is what the `unordered_set` stores.

```cpp
class Solution {
public:
    int minOperations(vector<int>& nums, vector<int>& target) {
        int N = nums.size();
        unordered_set<int> ans;
        for (int i = 0; i < N; ++i) {
            if (nums[i] != target[i]) ans.emplace(nums[i]);
        }
        return ans.size();
    }
};
```

## Q3. Number of Alternating XOR Partitions

### Solution 1: prefix xor, dynamic programming, hash map

Let `pref` be the xor of the prefix processed so far. If the last block of a partition ends here and must have xor `target1`, then after cutting at some earlier prefix xor `p` we need `p ^ pref = target1`, so `p = pref ^ target1`. The same idea gives `pref ^ target2` when the last block should be `target2`.

The maps store exactly those counts:

- `dp1[x]`: number of valid partitions of a processed prefix whose total prefix xor is `x` and whose last block xor is `target1`
- `dp2[x]`: the same, but the last block xor is `target2`

So `ans1 = dp2[pref ^ target1]` counts ways to append a `target1` block after a partition that previously ended with `target2`, and `ans2 = dp1[pref ^ target2]` does the symmetric transition. Seeding `dp2[0] = 1` treats the empty prefix as a virtual `target2` state, which lets the first real block start with `target1` as required.

```cpp
const int MOD = 1e9 + 7;
class Solution {
public:
    int alternatingXOR(vector<int>& nums, int target1, int target2) {
        int N = nums.size();
        map<int, int> dp1, dp2;
        dp2[0] = 1;
        int pref = 0, ans1 = 0, ans2 = 0;
        for (int val : nums) {
            pref ^= val;
            int x = pref ^ target1, y = pref ^ target2;
            ans1 = dp2[x], ans2 = dp1[y];
            dp1[pref] = (dp1[pref] + ans1) % MOD;
            dp2[pref] = (dp2[pref] + ans2) % MOD;
        }
        return (ans1 + ans2) % MOD;
    }
};
```

## Q4. Minimum Edge Toggles on a Tree

### Solution 1: tree DFS, postorder greedy, parity

Process the tree bottom-up. After we finish a child subtree, every node below that child is already settled, so if node `v` still has `s[v] != t[v]`, the only remaining edge that can fix `v` is the edge from `v` to its parent.

That makes the greedy choice forced: toggle that parent edge immediately. It flips `v` into the correct state and also flips the parent, which is fine because the parent has not been finalized yet. This is why a postorder DFS is the natural order.

In the code, `dfs(v)` first solves all descendants of `v`. If `s[v] != t[v]` afterward, we record the edge index, flip both endpoints with `s[v] ^= 1` and `s[u] ^= 1`, and continue upward. If the root still mismatches at the end, there is no parent edge left to fix it, so the answer is impossible.

```cpp
class Solution {
private:
    vector<vector<pair<int, int>>> adj;
    vector<int> ans, s, t;
    void dfs(int u, int p = -1) {
        for (const auto &[v, i] : adj[u]) {
            if (v == p) continue;
            dfs(v, u);
            if (s[v] != t[v]) {
                ans.emplace_back(i);
                s[v] ^= 1;
                s[u] ^= 1;
            }
        }
    }
public:
    vector<int> minimumFlips(int n, vector<vector<int>>& edges, string start, string target) {
        adj.assign(n, vector<pair<int, int>>());
        for (int i = 0; i + 1 < n; ++i) {
            int u = edges[i][0], v = edges[i][1];
            adj[u].emplace_back(v, i);
            adj[v].emplace_back(u, i);
        }
        s.assign(n, 0);
        t.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            s[i] = start[i] - '0';
            t[i] = target[i] - '0';
        }
        dfs(0);
        for (int i = 0; i < n; ++i) {
            if (s[i] != t[i]) return {-1};
        }
        sort(ans.begin(), ans.end());
        return ans;
    }
};
```

# Leetcode Weekly Contest 483

## Q1. Largest Even Number

### Solution 1: greedy suffix trimming

This implementation keeps deleting characters from the end until the last character is `'2'`, then returns the remaining prefix.

```cpp
class Solution {
public:
    string largestEven(string s) {
        while (!s.empty() && s.back() != '2') s.pop_back();
        return s;
    }
};
```

## Q2. Word Squares II

### Solution 1: brute force over 4-tuples, direct corner validation

The code sorts the words, then tries every ordered choice of four distinct words to play the roles:

- top
- left
- right
- bottom

For each quadruple, `isValid` checks the four corner constraints of the square:

- top-left must match between `top` and `left`
- top-right must match between `top` and `right`
- bottom-left must match between `bottom` and `left`
- bottom-right must match between `bottom` and `right`

If all four corner characters match, that quadruple is added to the answer.

So this is a straightforward exhaustive search with `O(N^4)` combinations, with only `O(1)` work to validate each candidate.

```cpp
class Solution {
private:
    bool isValid(const string& top, const string& left, const string& right, const string& bot) {
        return top[0] == left[0] && top[3] == right[0] && bot[0] == left[3] && bot[3] == right[3];
    }
public:
    vector<vector<string>> wordSquares(vector<string>& words) {
        int N = words.size();
        sort(words.begin(), words.end());
        vector<vector<string>> ans;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                for (int k = 0; k < N; ++k) {
                    if (k == j || k == i) continue;
                    for (int l = 0; l < N; ++l) {
                        if (l == k || l == j || l == i) continue;
                        if (isValid(words[i], words[j], words[k], words[l])) {
                            vector<string> cur = {words[i], words[j], words[k], words[l]};
                            ans.emplace_back(cur);
                        }
                    }
                }
            }
        }
        return ans;
    }
};
```

## Q3. Minimum Cost to Make Two Binary Strings Equal

### Solution 1: count mismatches by type, then greedily trade flips for paired operations

The first observation is that only mismatched positions matter. The code splits them into two buckets:

- `zero`: positions where `s[i] = 0` and `t[i] = 1`
- `one`: positions where `s[i] = 1` and `t[i] = 0`

If we fix every mismatch independently, the starting cost is just

`flipCost * (zero + one)`.

From there, the loop repeatedly replaces two individual flips with a possibly cheaper paired operation:

- if we have one mismatch of each type, a swap can fix both together, so we consume one from each bucket and add `swapCost`
- otherwise we take two mismatches of the same type, pay `crossCost + swapCost`, and again remove two pending flips

Since replacing two flips removes cost `2 * flipCost`, the running total is updated by adding the paired-operation cost and subtracting those two flips. The answer is the minimum value seen during this process.

So the algorithm is a greedy sweep over how many pairs we decide to resolve using swap-style operations instead of plain flips.

```cpp
using int64 = long long;
const int64 INF = numeric_limits<int64>::max();
class Solution {
public:
    int64 minimumCost(string s, string t, int flipCost, int swapCost, int crossCost) {
        int64 ans = INF;
        int zero = 0, one = 0, N = s.size();
        for (int i = 0; i < N; ++i) {
            if (s[i] != t[i]) {
                if (s[i] == '0') zero++;
                else one++;
            }
        }
        // start with most flips
        int64 cur = 1LL * flipCost * (zero + one);
        ans = min(ans, cur);
        while (one + zero > 1) {
            if (zero > 0 && one > 0) {
                zero--;
                one--;
                cur += swapCost;
            } else {
                if (one > 0) one -= 2;
                else zero -= 2;
                cur += crossCost;
                cur += swapCost;
            }
            cur -= 2LL * flipCost;
            ans = min(ans, cur);
        }
        return ans;
    }
};
```

## Q4. Minimum Cost to Merge Sorted Lists

### Solution 1: subset DP, precompute merged list statistics for every mask

This solution treats each subset of lists as one state.

For every bitmask `mask`, it precomputes:

- `arr[mask]`: the fully merged sorted list of all lists in that subset
- `len[mask]`: the size of that merged list
- `med[mask]`: the median of that merged list

Those values are built incrementally by removing the least significant set bit and merging the previously built subset with one more list.

Once that preprocessing is done, `dp[mask]` means the minimum cost to merge all lists inside `mask` into one list. The transition enumerates every non-empty proper submask:

- split `mask` into `submask` and `mask ^ submask`
- optimally merge both halves first
- then pay the final combine cost

That final cost is

`len[left] + len[right] + abs(med[left] - med[right])`.

So this is a classic subset DP over all partitions of each mask. It is exponential, but it matches the structure of the recurrence very directly.

```cpp
using int64 = long long;
const int64 INF = numeric_limits<int64>::max();
class Solution {
private:
    int median(const vector<int>& arr) {
        int N = arr.size();
        int m = (N - 1) / 2;
        return arr[m];
    }
    vector<int> merge(const vector<int>& A, const vector<int>& B) {
        int N = A.size(), M = B.size();
        vector<int> arr;
        arr.reserve(N + M);
        for (int i = 0, j = 0; i < N || j < M;) {
            if (i == N) {
                arr.emplace_back(B[j++]);
            } else if (j == M) {
                arr.emplace_back(A[i++]);
            } else if (A[i] <= B[j]) {
                arr.emplace_back(A[i++]);
            } else {
                arr.emplace_back(B[j++]);
            }
        }
        return arr;
    }
public:
    int64 minMergeCost(vector<vector<int>>& lists) {
        int N = lists.size();
        int endMask = 1 << N;
        vector<int> len(endMask, 0), med(endMask, 0);
        vector<vector<int>> arr(endMask, vector<int>());
        for (int mask = 1; mask < endMask; ++mask) {
            int lsb = __builtin_ctz(mask);
            if ((mask & (mask - 1)) == 0) { // singular
                arr[mask] = lists[lsb];
            } else {
                arr[mask] = merge(arr[mask ^ (1 << lsb)], lists[lsb]);
            }
            // find smallest set bit
            med[mask] = median(arr[mask]);
            len[mask] = arr[mask].size();
        }
        vector<int64> dp(endMask, INF);
        dp[0] = 0;
        for (int mask = 1; mask < endMask; ++mask) {
            if ((mask & (mask - 1)) == 0) {
                dp[mask] = 0;
                continue;
            }
            for (int submask = mask - 1; submask > 0; submask = (submask - 1) & mask) {
                int submask2 = mask ^ submask;
                int64 cost = dp[submask] + dp[submask2] + len[submask] + len[submask2] + abs(med[submask] - med[submask2]);
                dp[mask] = min(dp[mask], cost);
            }
        }
        return dp.back();
    }
};
```

# Leetcode Weekly Contest 490

Q1: Follow the rules, toggling a boolean variable to determine whether to add or subtract the current number. 
Q2: brute force permutations of the digits, and precompute the factorial of digits.
Q3: Greedily place as many 1s as possible in locations where s has 0s, prioritizing leftmost locations, and then fill in the remaining 1s from the right.
Q4: sparse dynamic programming with a map to store the count of sequences that give a specific pair of numerator and denominator to represent rational numbers.

## Find the Score Difference in a Game

### Solution 1: bit logic, toggling, remainder, parity

```cpp
class Solution {
public:
    int scoreDifference(vector<int>& nums) {
        int ans = 0, N = nums.size();
        bool active = true;
        for (int i = 0; i < N; ++i) {
            if (nums[i] & 1) active ^= true;
            if ((i + 1) % 6 == 0) active ^= true;
            if (active) ans += nums[i];
            else ans -= nums[i];
        }
        return ans;
    }
};
```

## Check Digitorial Permutation

### Solution 1: permutations, factorial, string manipulation

```cpp
class Solution {
private:
    vector<int> fact;
    bool isDigitorial(const string& num) {
        if (num[0] == '0') return false;
        int val = accumulate(num.begin(), num.end(), 0, [&](int accum, char x) { return accum + fact[x - '0']; });
        return val == stoi(num);
    }
public:
    bool isDigitorialPermutation(int n) {
        fact.assign(10, 1);
        for (int i = 2; i < 10; ++i) {
            for (int j = 2; j <= i; ++j) {
                fact[i] *= j;
            }
        }
        string num = to_string(n);
        sort(num.begin(), num.end());
        do {
            if (isDigitorial(num)) return true;
        } while (next_permutation(num.begin(), num.end()));
        return false;
    }
};
```

## Maximum Bitwise XOR After Rearrangement

### Solution 1: string manipulation, bitwise XOR, greedy

```cpp
class Solution {
public:
    string maximumXor(string s, string t) {
        int N = s.size();
        string u = string(N, '0');
        int rem = count_if(t.begin(), t.end(), [](const char& ch) { return ch == '1'; });
        for (int i = 0; i < N && rem > 0; ++i) {
            if (s[i] == '0') {
                u[i] = '1';
                --rem;
            }
        }
        for (int i = N - 1; i >= 0 && rem > 0; --i) {
            if (u[i] == '0') {
                u[i] = '1';
                --rem;
            }
        }
        string ans = string(N, '0');
        for (int i = 0; i < N; ++i) {
            if (s[i] != u[i]) ans[i] = '1';
        }
        return ans;
    }
};
```

## Count Sequences to K

### Solution 1: dynamic programming, map, pair of products, counting sequences

```cpp
using int64 = long long;
class Solution {
public:
    int countSequences(vector<int>& nums, int64 k) {
        map<pair<int64, int64>, int> dp, ndp;
        dp[{1, 1}] = 1;
        int N = nums.size();
        int64 x, y;
        for (int i = 0; i < N; ++i) {
            ndp.clear();
            for (const auto &[p, v] : dp) {
                tie(x, y) = p;
                // multiply
                ndp[{x * nums[i], y}] += v;
                // divide
                ndp[{x, y * nums[i]}] += v;
                // unchanged
                ndp[{x, y}] += v;
            }
            swap(dp, ndp);
        }
        int ans = 0;
        for (const auto &[p, v] : dp) {
            if (p.first % p.second == 0 && p.first / p.second == k) ans += v;
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 177

## 3853. Merge Close Characters

### Solution 1: two pointers, greedy

```cpp
class Solution {
public:
    string mergeCharacters(string s, int k) {
        vector<int> last(26, -1000);
        string ans;
        int N = s.size();
        for (int i = 0, j = 0; i < N; ++i) {
            int idx = s[i] - 'a';
            if (j - last[idx] > k) {
                ans += s[i];
                last[idx] = j;
                ++j;
            }
        }
        return ans;
    }
};
```

## 3854. Minimum Operations to Make Array Parity Alternating

### Solution 1: feasibility check, combinations

Just try all possible combinations for min and max value, cause there are only like 4 of them. cause you can only increase or decrease the min and max by one value.

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
private:
    vector<int> calc1(vector<int> nums, int tmin, int tmax) {
        int minVal = *min_element(nums.begin(), nums.end()) + tmin;
        int maxVal = *max_element(nums.begin(), nums.end()) + tmax;
        int N = nums.size();
        vector<int> ans(2, 0);
        for (int i = 0; i < N; ++i) {
            if (i % 2 == 0 && nums[i] & 1) {
                ans[0]++;
                if (nums[i] <= minVal) nums[i]++;
                else nums[i]--;
            } else if (i & 1 && nums[i] % 2 == 0) {
                ans[0]++;
                if (nums[i] <= minVal) nums[i]++;
                else nums[i]--;
            }
        }
        ans[1] = *max_element(nums.begin(), nums.end()) - *min_element(nums.begin(), nums.end());
        return ans;
    }
    vector<int> calc2(vector<int> nums, int tmin, int tmax) {
        int minVal = *min_element(nums.begin(), nums.end()) + tmin;
        int maxVal = *max_element(nums.begin(), nums.end()) + tmax;
        int N = nums.size();
        vector<int> ans(2, 0);
        for (int i = 0; i < N; ++i) {
            if (i & 1 && nums[i] & 1) {
                ans[0]++;
                if (nums[i] <= minVal) nums[i]++;
                else nums[i]--;
            } else if (i % 2 == 0 && nums[i] % 2 == 0) {
                ans[0]++;
                if (nums[i] <= minVal) nums[i]++;
                else nums[i]--;
            }
        }
        ans[1] = *max_element(nums.begin(), nums.end()) - *min_element(nums.begin(), nums.end());
        return ans;
    }
public:
    vector<int> makeParityAlternating(vector<int>& nums) {
        vector<int> ans = {INF, INF};
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                ans = min({ans, calc1(nums, i, j), calc2(nums, i, j)});
            }
        }
        return ans;
    }
};
```

## 3855. Sum of K-Digit Numbers in a Range

### Solution 1: binary exponentiation, mathematics, combinatorics

The main part I was stuck on was how to calculate the sum of 10^i from 0 to k - 1, which there is a trick to just take 10^k - 1 and divide by 9, which is the sum of a geometric series.

```cpp
using int64 = long long;
const int MOD = 1e9 + 7;
class Solution {
private:
    int64 inv(int i, int64 m) {
        return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
    }
    int64 exponentiation(int64 b, int64 p, int64 m) {
        int64 res = 1;
        while (p > 0) {
            if (p & 1) res = (res * b) % m;
            b = (b * b) % m;
            p >>= 1;
        }
        return res;
    }
    int sum(int l, int r) {
        int ans = 0;
        for (int i = l; i <= r; ++i) {
            ans += i;
        }
        return ans;
    }
public:
    int sumOfNumbers(int l, int r, int k) {
        int64 ans = sum(l, r);
        ans = ans * exponentiation(r - l + 1, k - 1, MOD) % MOD;
        int64 term = exponentiation(10, k, MOD) - 1;
        term = term * inv(9, MOD) % MOD;
        if (term < 0) term += MOD;
        ans = ans * term % MOD;
        return ans;
    }
};
```

# Leetcode Weekly Contest 491

Q1: pop characters from back until reach non-vowel character
Q2: The cost is to always split off 1 and n - 1, recursively, so the answer is just sum of 1 + 2 + 3 ... + n - 1, which is n * (n - 1) / 2.
Q3: Greedily block bits from the highest bit to the lowest bit, and check if blocking that bit is needed by checking if there is a row that has all 1s in that bit. If it is needed, then we keep that bit in the answer, otherwise we can block that bit by setting it to 0 in the grid.
Q4: Two sliding windows, one to maintain at most k distinct integers, and another to maintain k distinct integers with at least m frequency, and count the number of valid subarrays.  Used two frequency maps to maintain the couunts of both windows. The overlap between the windows for each endpoint i gives the number of valid subarrays ending at i.

## Q1. Trim Trailing Vowels

### Solution 1: string manipulation, string find, string pop_back

```cpp
class Solution {
public:
    string trimTrailingVowels(string s) {
        string vowels = "aeiou";
        int N = s.size();
        while (!s.empty() && vowels.find(s.back()) != string::npos) {
            s.pop_back();
        }
        return s;
    }
};
```

## Q2. Minimum Cost to Split into Ones

### Solution 1: counting

```cpp
class Solution {
public:
    int minCost(int n) {
        return n * (n - 1) / 2;
    }
};
```

## Q3. Minimum Bitwise OR From Grid

### Solution 1: greedy, bit manipulation, in-place modification

```cpp
class Solution {
public:
    int minimumOR(vector<vector<int>>& grid) {
        int R = grid.size(), C = grid[0].size();
        int ans = 0;
        for (int i = 20; i >= 0; --i) {
            bool needed = false;
            for (int r = 0; r < R; ++r) {
                int total = 0, cnt = 0;
                for (int c = 0; c < C; ++c) {
                    if ((grid[r][c] >> i) & 1) {
                        cnt++;
                    }
                    if (grid[r][c]) total++;
                }
                if (cnt == total) {
                    needed = true;
                    break;
                }
            }
            if (needed) {
                ans |= (1 << i);
                continue;
            }
            for (int r = 0; r < R; ++r) {
                for (int c = 0; c < C; ++c) {
                    if ((grid[r][c] >> i) & 1) {
                        grid[r][c] = 0; // can block
                    }
                }
            }
        }
        return ans;
    }
};
```

## Q4. Count Subarrays With K Distinct Integers

### Solution 1: sliding window, two pointers, map, counting distinct integers, counting integers with frequency m

```cpp
using int64 = long long;
class Solution {
public:
    int64 countSubarrays(vector<int>& nums, int k, int m) {
        int N = nums.size();
        map<int, int> freq, f2;
        int cntK = 0, cntM = 0;
        int64 ans = 0;
        for (int i = 0, l = 0, r = 0; i < N; ++i) {
            freq[nums[i]]++;
            f2[nums[i]]++;
            if (freq[nums[i]] == 1) {
                cntK++;
            }
            if (f2[nums[i]] == m) {
                cntM++;
            }
            while (cntK > k) {
                freq[nums[l]]--;
                if (freq[nums[l]] == 0) cntK--;
                l++;
            }
            while (r < l || cntM > k  || (cntM == k && f2[nums[r]] > m)) {
                f2[nums[r]]--;
                if (f2[nums[r]] == m - 1) cntM--;
                r++;
            }
            if (cntK == k && cntM == k) {
                ans += r - l + 1;
            }
        }
        return ans;
    }
};
```

# Leetcode Weekly Contest 492

Q1: greedy, find smallest index with smallest best capacity that can fit the item, and return that index.
Q2: prefix sum and suffix product, check if equal at any index
Q3: Always best to take the substring that will only be missing either the first or last character, so perform analysis based on if the first and last character is going to move when sorting, cause that has implications for number of operations needed.
Q4: Divide and Conquer recursion, Calculate the minimum cost for any segment, and if it is better to break into two pieces then use that for that segment.

##

### Solution 1: greedy

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int minimumIndex(vector<int>& capacity, int itemSize) {
        int N = capacity.size();
        int best = INF, idx = -1;
        for (int i = 0; i < N; ++i) {
            if (capacity[i] < itemSize) continue;
            if (capacity[i] < best) {
                best = capacity[i];
                idx = i;
            }
        }
        return idx;
    }
};
```

##

### Solution 1: prefix sum, suffix product, check for equality

```cpp
using int64 = long long;
const int64 INF = 1e15;
class Solution {
public:
    int smallestBalancedIndex(vector<int>& nums) {
        int ans = -1, N = nums.size();
        __int128 suf = 1;
        int64 pref = accumulate(nums.begin(), nums.end(), 0LL);
        for (int i = N - 1; i >= 0; --i) {
            pref -= nums[i];
            if (pref == suf) ans = i;
            suf *= nums[i];
            if (suf > INF) break;
        }
        return ans;
    }
};
```

##

### Solution 1: edge cases, string manipulation, sorting, counting operations

```cpp
class Solution {
public:
    int minOperations(string s) {
        int N = s.size();
        char minc = *min_element(s.begin(), s.end());
        char maxc = *max_element(s.begin(), s.end());
        int cntmin = count_if(s.begin(), s.end(), [&](char ch) { return ch == minc; });
        int cntmax = count_if(s.begin(), s.end(), [&](char ch) { return ch == maxc; });
        string ss = s;
        sort(ss.begin(), ss.end());
        if (s == ss) return 0;
        if (N == 2) return -1;
        if (s[0] == maxc && s.back() == minc && cntmin == 1 && cntmax == 1) return 3;
        if (s[0] == minc || s.back() == maxc) return 1;
        return 2;
    }
};
```

##

### Solution 1: divide and conquer, recursion, minimize, optimization

```cpp
using int64 = long long;
class Solution {
private:
    int64 dfs(int i, int j, const string &s, int encCost, int flatCost) {
        int N = j - i + 1;
        int X = 0;
        for (int k = i; k <= j; ++k) {
            if (s[k] == '1') X++;
        }
        int64 ans = X == 0 ? flatCost : 1LL * N * X * encCost;
        if (N & 1) return ans;
        int m = i + (j - i) / 2;
        int64 x1 = dfs(i, m, s, encCost, flatCost);
        int64 x2 = dfs(m + 1, j, s, encCost, flatCost);
        return min(ans, x1 + x2);
    }
public:
    int64 minCost(string s, int encCost, int flatCost) {
        int N = s.size();
        return dfs(0, N - 1, s, encCost, flatCost);   
    }
};
```

## 1622. Fancy Sequence

### Solution 1: distributive property, order of operations, math, arithmetic, modular inverse, solving algebraic equations

Draw the array, and think about how can I solve for the a in this interval, given if I know a at the idx and the current endpoint
And same of the other value which is b, b stores the cumulative addition and multiplication that are together. 

And the other stores just the cumulative multiplication, which is all you need. 

```cpp
using int64 = long long;
const int MOD = 1e9 + 7;
int64 inv(int i, int64 m) {
    return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}
class Fancy {
private:
    vector<int> arr, vals, mult;
public:
    Fancy() {
        vals.emplace_back(0);
        mult.emplace_back(1);
    }
    
    void append(int val) {
        arr.emplace_back(val);
        vals.emplace_back(vals.back());
        mult.emplace_back(mult.back());
    }
    
    void addAll(int inc) {
        vals.back() += inc;
        vals.back() %= MOD;
    }
    
    void multAll(int m) {
        vals.back() = 1LL * vals.back() * m % MOD;
        mult.back() = 1LL * mult.back() * m % MOD;
    }
    
    int getIndex(int idx) {
        if (idx >= arr.size()) {
            return -1;
        }
        int a = 1LL * mult.back() * inv(mult[idx], MOD) % MOD;
        int b = (vals.back() - 1LL * a * vals[idx] % MOD + MOD) % MOD;
        int ans = (b + 1LL * arr[idx] * a % MOD) % MOD; 
        return ans;
    }
};
```

# Leetcode Biweekly Contest 178

## Q1. First Unique Even Element

### Solution 1: map, frequency counting, iteration

```cpp
class Solution {
public:
    int firstUniqueEven(vector<int>& nums) {
        map<int, int> freq;
        for (int x : nums) freq[x]++;
        for (int x : nums) {
            if (freq[x] > 1) continue;
            if (x % 2 == 0) return x;
        }
        return -1;
    }
};
```

## Q2. Sum of GCD of Formed Pairs

### Solution 1: prefix max, sorting, gcd

```cpp
using int64 = long long;
class Solution {
public:
    int64 gcdSum(vector<int>& nums) {
        int N = nums.size();
        vector<int> pref(N, 0);
        int pmax = 0;
        for (int i = 0; i < N; ++i) {
            pmax = max(pmax, nums[i]);
            pref[i] = gcd(nums[i], pmax);
        }
        sort(pref.begin(), pref.end());
        int64 ans = 0;
        for (int i = 0, j = N - 1; i < j; ++i, --j) {
            ans += gcd(pref[i], pref[j]);
        }
        return ans;
    }
};
```

## Q3. Minimum Cost to Equalize Arrays Using Swaps

### Solution 1: set, map, frequency counting, greedy

```cpp
class Solution {
public:
    int minCost(vector<int>& nums1, vector<int>& nums2) {
        int N = nums1.size();
        unordered_set<int> nums;
        unordered_map<int, int> freq1, freq2;
        for (int x : nums1) {
            nums.emplace(x);
            freq1[x]++;
        }
        for (int x : nums2) {
            nums.emplace(x);
            freq2[x]++;
        }
        for (int x : nums) {
            int tot = freq1[x] + freq2[x];
            if (tot & 1) return -1;
        }
        int ans = 0;
        for (const auto &[k, v] : freq1) {
            int tot = v + freq2[k];
            ans += max(0, v - tot / 2);
        }
        return ans;
    }
};
```

## Q4. Count Fancy Numbers in a Range

### Solution 1: digit dp

```cpp
using int64 = long long;
class Solution {
private:
    int64 dp[16][140][10][4][2][2];
    bool isIncreasing(int x) {
        int prv = -1;
        while (x > 0) {
            int d = x % 10;
            if (d <= prv) return false;
            prv = d;
            x /= 10;
        }
        return true;
    }
    bool isDecreasing(int x) {
        int prv = 10;
        while (x > 0) {
            int d = x % 10;
            if (d >= prv) return false;
            prv = d;
            x /= 10;
        }
        return true;
    }
    bool isGood(int x) {
        return isDecreasing(x) || isIncreasing(x);
    }
    int64 dfs(const string& s, int idx, int sumOfDig, int lastDig, int status, int tight, int zero) {
        int N = s.size();
        if (idx == N) {
            if (zero) return 0;
            if (status != 3) return 1;
            if (isGood(sumOfDig)) return 1;
            return 0;
        }
        if (dp[idx][sumOfDig][lastDig][status][tight][zero] != -1) return dp[idx][sumOfDig][lastDig][status][tight][zero];
        int64 ans = 0;
        int curDig = s[idx] - '0';
        for (int dig = 0; dig < 10; ++dig) {
            if (tight && dig > curDig) break;
            if (zero) {
                ans += dfs(s, idx + 1, sumOfDig + dig, dig, status, tight && curDig == dig, zero && dig == 0);
            } else if (status == 0) {
                ans += dfs(s, idx + 1, sumOfDig + dig, dig, dig == lastDig ? 3 : dig < lastDig ? 1 : 2, tight && curDig == dig, zero && dig == 0);
            } else if (status == 1) {
                ans += dfs(s, idx + 1, sumOfDig + dig, dig, dig < lastDig ? status : 3, tight && curDig == dig, zero && dig == 0);
            } else if (status == 2) {
                ans += dfs(s, idx + 1, sumOfDig + dig, dig, dig > lastDig ? status : 3, tight && curDig == dig, zero && dig == 0);
            } else if (status == 3) {
                ans += dfs(s, idx + 1, sumOfDig + dig, dig, status, tight && curDig == dig, zero && dig == 0);
            }
        }
        return dp[idx][sumOfDig][lastDig][status][tight][zero] = ans;
    }
    int64 calc(int64 x) {
        fill(&dp[0][0][0][0][0][0], &dp[0][0][0][0][0][0] + 16 * 140 * 10 * 4 * 2 * 2, -1LL);
        return dfs(to_string(x), 0, 0, 0, 0, 1, 1);
    }
public:
    int64 countFancy(int64 l, int64 r) {
        return calc(r) - calc(l - 1);
    }
};
```

# Leetcode Weekly Contest 493

## Q2. Count Commas in Range II

### Solution 1: place value counting, powers of 1000

Every number contributes one comma for each comma boundary it passes in standard formatting:

- numbers in `[1,000, 999,999]` contribute at least 1 comma
- numbers in `[1,000,000, 999,999,999]` contribute at least 2 commas
- and so on

So instead of formatting every number from `1` to `n`, we count how many numbers are at least `10^3`, how many are at least `10^6`, how many are at least `10^9`, etc.

If `base = 1000^k`, then exactly `n - base + 1` numbers in `[1, n]` contribute their `k`-th comma. Summing this over all valid comma positions gives the answer in `O(log_1000 n)` time.

```cpp
using int64 = long long;
class Solution {
public:
    int64 countCommas(int64 n) {
        int64 base = 1e3, ans = 0;
        while (base <= n) {
            ans += n - base + 1;
            base *= 1e3;
        }
        return ans;
    }
};
```

## Q3. Longest Arithmetic Sequence After Changing At Most One Element

### Solution 1: prefix and suffix runs of arithmetic sequences, difference array, casework with longest runs of arithmetic sequences.

For an arithmetic subarray, the adjacent differences must all be equal, so the first step is to build the difference array `diff[i] = nums[i + 1] - nums[i]`.

Then:

- `pref[i]` stores the length of the longest run of equal differences ending at `diff[i]`
- `suf[i]` stores the length of the longest run of equal differences starting at `diff[i]`

This immediately gives the best answer without changing anything. Since we are allowed to modify at most one element, we can also extend an existing arithmetic run by one more position, which is why the code does `if (ans < N) ans++`.

The interesting case is changing some middle element `nums[i]`. To make `nums[i - 1], nums[i], nums[i + 1]` arithmetic, the new middle value must be the average of its neighbors, so `nums[i + 1] - nums[i - 1]` must be even. If that works, the common difference is

`delta = (nums[i + 1] - nums[i - 1]) / 2`.

Now we check how far an arithmetic run with difference `delta` already extends to the left and right using `pref` and `suf`, and merge them through the changed middle element. That gives a candidate length of `leftRun + rightRun + 3`.

Overall this is linear time after building the difference array.

```cpp
class Solution {
public:
    int longestArithmetic(vector<int>& nums) {
        int N = nums.size();
        int M = N - 1;
        vector<int> diff(M, 0), pref(N, 1), suf(N, 1);
        for (int i = 1; i < N; ++i) {
            diff[i - 1] = nums[i] - nums[i - 1];
        }
        for (int i = 1; i < M; ++i) {
            if (diff[i - 1] == diff[i]) pref[i] = pref[i - 1] + 1;
        }
        for (int i = M - 2; i >= 0; --i) {
            if (diff[i] == diff[i + 1]) suf[i] = suf[i + 1] + 1;
        }
        int ans = max(*max_element(pref.begin(), pref.end()), *max_element(suf.begin(), suf.end())) + 1;
        if (ans < N) ans++;
        for (int i = 1; i + 1 < N; ++i) {
            if ((nums[i + 1] - nums[i - 1]) % 2 != 0) continue;
            int delta = (nums[i + 1] - nums[i - 1]) / 2;
            int leftRun = 0;
            if (i - 2 >= 0 && diff[i - 2] == delta) {
                leftRun = pref[i - 2];
            }
            int rightRun = 0;
            if (i + 1 < M && diff[i + 1] == delta) {
                rightRun = suf[i + 1];
            }
            ans = max(ans, leftRun + rightRun + 3);
        }
        return ans;
    }
};
```

## Q4. Maximum Points Activated with One Addition

### Solution 1: union find, map, grouping, counting, sorting

Treat each point as a node in a graph. Two points belong to the same connected component if they share an `x` coordinate or a `y` coordinate, either directly or through a chain of such connections.

The DSU builds these components efficiently:

- group indices by `x`, and union adjacent indices inside each `x` bucket
- group indices by `y`, and union adjacent indices inside each `y` bucket

After that, `groups()` returns the size of every connected component.

When we add one new point `(x, y)`, it can connect to all existing points on row `y` and all existing points on column `x`. Because points sharing the same row are already in one component, and points sharing the same column are already in one component, one added point can merge at most two existing components:

- it may attach to only one component, giving `largest + 1`
- or it may bridge two different components, giving `largest + secondLargest + 1`

So once the component sizes are known, the answer is just the better of those two cases.

```cpp
struct UnionFind {
    vector<int> parent, size;
    UnionFind(int n) {
        parent.resize(n);
        iota(parent.begin(),parent.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i == parent[i]) {
            return i;
        }
        return parent[i] = find(parent[i]);
    }

    void unite(int i, int j) {
        i = find(i), j = find(j);
        if (i != j) {
            if (size[j] > size[i]) {
                swap(i, j);
            }
            size[i] += size[j];
            parent[j] = i;
        }
    }

    bool same(int i, int j) {
        return find(i) == find(j);
    }

    vector<int> groups() {
        int n = parent.size();
        unordered_map<int, int> group_map;
        for (int i = 0; i < n; ++i) {
            group_map[find(i)]++;
        }
        vector<int> res;
        for (const auto &[k, v] : group_map) {
            res.emplace_back(v);
        }
        return res;
    }
};
class Solution {
public:
    int maxActivated(vector<vector<int>>& points) {
        int N = points.size();
        UnionFind dsu(N);
        map<int, vector<int>> xcoord, ycoord;
        for (int i = 0; i < N; ++i) {
            int x = points[i][0], y = points[i][1];
            xcoord[x].emplace_back(i);
            ycoord[y].emplace_back(i);
        }
        for (const auto &[k, vals] : xcoord) {
            for (int i = 1; i < vals.size(); ++i) {
                dsu.unite(vals[i - 1], vals[i]);
            }
        }
        for (const auto &[k, vals] : ycoord) {
            for (int i = 1; i < vals.size(); ++i) {
                dsu.unite(vals[i - 1], vals[i]);
            }
        }
        vector<int> g = dsu.groups();
        sort(g.begin(), g.end());
        int ans = g.back() + 1;
        if (g.size() > 1) ans = max(ans, g.end()[-1] + g.end()[-2] + 1);
        return ans;
    }
};
```

## 3130. Find All Possible Stable Binary Arrays II

### Solution 1: dynamic programming, counting, combinatorics, prefix sums for optimization

Let `dp[i][j][k]` be the number of stable arrays that use exactly `i` zeros and `j` ones and end with digit `k`.

The natural recurrence is:

- to end with `0`, append a block of zeros to a valid array that previously ended with `1`
- to end with `1`, append a block of ones to a valid array that previously ended with `0`

The stability rule says no block can have length greater than `limit`. A direct transition would try every possible last block length from `1` to `limit`, but that would be too slow.

The code rewrites that sum into a sliding-window style recurrence:

- `dp[i][j][0]` starts from all arrays obtained by appending one more `0`
- if that creates runs longer than `limit`, subtract the states that would force an illegal block
- do the symmetric transition for `dp[i][j][1]`

That subtraction is the optimization that removes the extra `limit` factor, so the DP runs in `O(zero * one)` time instead of `O(zero * one * limit)`.

```cpp
const int MOD = 1e9 + 7;
class Solution {
public:
    int numberOfStableArrays(int zero, int one, int limit) {
        vector<vector<vector<int>>> dp(zero + 1, vector<vector<int>>(one + 1, vector<int>(2, 0)));
        for (int i = 0; i <= zero; ++i) {
            for (int j = 0; j <= one; ++j) {
                for (int k = 0; k <= 1; ++k) {
                    if (i == 0) { // no zero, adding 1, if j is less than limit, then you can do it one way
                        if (k == 1 && j <= limit) {
                            dp[i][j][k] = 1;
                        }
                    } else if (j == 0) {
                        if (k == 0 && i <= limit) {
                            dp[i][j][k] = 1;
                        }
                    } else if (k == 0) {
                        dp[i][j][k] = (dp[i - 1][j][0] + dp[i - 1][j][1]) % MOD;
                        if (i > limit) {
                            dp[i][j][k] -= dp[i - limit - 1][j][1];
                        }
                    } else { // k = 1
                        dp[i][j][k] = (dp[i][j - 1][0] + dp[i][j - 1][1]) % MOD;
                        if (j > limit) {
                            dp[i][j][k] -= dp[i][j - limit - 1][0];
                        }
                    }
                    if (dp[i][j][k] < 0) dp[i][j][k] += MOD;
                }
            }
        }
        int ans = (dp[zero][one][0] + dp[zero][one][1]) % MOD;
        return ans;
    }
};
```

# Leetcode Weekly Contest 494

## Q2. Construct Uniform Parity Array II

### Solution 1: min element, parity

if the smallest element is odd it is always possible, otherwise all elements must be even to be able to make them all the same parity.

```cpp
class Solution {
public:
    bool uniformArray(vector<int>& nums1) {
        int a = *min_element(nums1.begin(), nums1.end());
        if (a & 1) return true;
        return all_of(nums1.begin(), nums1.end(), [&](int x) { return x % 2 == 0; });
    }
};
```

## Q3. Minimum Removals to Achieve Target XOR

### Solution 1: subset sum with bitmasks, subset sum with xor operator, minimize the size of the subset that equals a target

Use the algebraic properties of xor operationn to compute the follownig
XOR(nums) ^ XOR(subset) = target
XOR(subset) = XOR(nums) ^ target
So yeah just solve for that new target, and find the subset that xors to it. 

```cpp
const int BIT = 14, INF = numeric_limits<int>::max();
class Solution {
public:
    int minRemovals(vector<int>& nums, int target) {
        int val = accumulate(nums.begin(), nums.end(), 0, [](int accum, int x) { return accum ^ x; });
        target ^= val;
        int endMask = 1 << BIT;
        vector<int> dp(endMask, INF), ndp(endMask, INF);
        dp[0] = 0;
        for (int x : nums) {
            for (int mask = 0; mask < endMask; ++mask) {
                ndp[mask] = dp[mask];
                int pmask = mask ^ x;
                if (pmask >= endMask) continue;
                if (dp[pmask] == INF) continue;
                ndp[mask] = min(ndp[mask], dp[pmask] + 1);
            }
            swap(dp, ndp);
        }
        int ans = dp[target];
        return ans < INF ? ans : -1;
    }
};
```

## Q4. Count Good Subarrays

### Solution 1: monotonic stack, counting subarrays, bit manipulation, combinatorics

You want to track the with monotonic stack the last index so that the current element is the largest element in that interval. 
And you also need to compute the lastBit, cause you need check for each bit that is 0 in current bit string, where it was last a 1, cause you have to clip your left or right endpoint based on that, and you want to take the max of all those last bits, cause you need to satisfy all of them.
Then the computation is easy by taking the number of ways to choose the left and right endpoint based on those constraints, and sum that for all i.

```cpp
using int64 = long long;
const int B = 31;
class Solution {
public:
    int64 countGoodSubarrays(vector<int>& nums) {
        int N = nums.size();
        vector<int> L(N, 0), R(N, N - 1), lastBit(B, -1);
        stack<int> stk, stk1;
        for (int i = 0; i < N; ++i) {
            while (!stk.empty() && nums[stk.top()] <= nums[i]) stk.pop();
            if (!stk.empty()) L[i] = stk.top() + 1;
            stk.emplace(i);
            for (int b = 0; b < 31; ++b) {
                if ((nums[i] >> b) & 1) {
                    lastBit[b] = i;
                    continue;
                };
                L[i] = max(L[i], lastBit[b] + 1);
            }
        }
        lastBit.assign(B, N);
        for (int i = N - 1; i >= 0; --i) {
            while (!stk1.empty() && nums[stk1.top()] < nums[i]) stk1.pop();
            if (!stk1.empty()) R[i] = stk1.top() - 1;
            stk1.emplace(i);
            for (int b = 0; b < 31; ++b) {
                if ((nums[i] >> b) & 1) {
                    lastBit[b] = i;
                    continue;
                }
                R[i] = min(R[i], lastBit[b] - 1);
            }
        }
        int64 ans = 0;
        for (int i = 0; i < N; ++i) {
            int l = L[i], r = R[i];
            ans += 1LL * (i - l + 1) * (r - i + 1);
        }
        return ans;
    }
};
```

## 2573. Find the String with LCP

### Solution 1: greedy, backtracking to check validity of lcp

```cpp
class Solution {
public:
    string findTheString(vector<vector<int>>& lcp) {
        int N = lcp.size();
        string ans(N, '#');
        char cur = 'a';
        for (int i = 0; i < N && cur; ++i) {
            if (ans[i] != '#') continue;
            if (cur > 'z') return "";
            ans[i] = cur;
            for (int j = i + 1; j < N; ++j) {
                if (lcp[i][j] > 0) {
                    ans[j] = ans[i];
                }
            }
            cur++;
        }
        for (int i = N - 1; i >= 0; --i) {
            for (int j = N - 1; j >= 0; --j) {
                if (ans[i] == ans[j]) {
                    if (i == N - 1 || j == N - 1) {
                        if (lcp[i][j] != 1) return "";
                    } else if (lcp[i][j] != lcp[i + 1][j + 1] + 1) return "";
                } else {
                    if (lcp[i][j] > 0) return "";
                }
            }
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 179

## Q1. Minimum Absolute Difference Between Two Values

### Solution 1: last seen

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int minAbsoluteDifference(vector<int>& nums) {
        int ans = INF, N = nums.size(), lastOne = -1, lastTwo = -1;
        for (int i = 0; i < N; ++i) {
            if (nums[i] == 1) {
                if (lastTwo != -1) ans = min(ans, i - lastTwo);
                lastOne = i;
            } else if (nums[i] == 2) {
                if (lastOne != -1) ans = min(ans, i - lastOne);
                lastTwo = i;
            }
        }
        return ans < INF ? ans : -1;
    }
};
```

## Q2. Direction Assignments with Exactly K Visible People

### Solution 1: combinatorics

```cpp
using int64 = long long;
const int MOD = 1e9 + 7;

int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

vector<int64> fact, inv_fact;

void factorials(int n, int64 m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1], m);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}

int64 choose(int n, int r, int64 m) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}

class Solution {
public:
    int countVisiblePeople(int n, int pos, int k) {
        factorials(n, MOD);
        int L = pos, R = n - pos - 1, ans = 0;
        for (int i = 0; i <= min(L, k); ++i) {
            int j = k - i; // number picking on right side
            if (R < j) continue;
            ans += 2LL * choose(L, i, MOD) * choose(R, j, MOD) % MOD;
            ans %= MOD;
        }
        return ans;
    }
};
```

## Q3. Minimum XOR Path in a Grid

### Solution 1: dynamic programming over xor states

```cpp
const int B = 10;
class Solution {
public:
    int minCost(vector<vector<int>>& grid) {
        int R = grid.size(), C = grid[0].size();
        vector<vector<vector<bool>>> dp(R, vector<vector<bool>>(C, vector<bool>(1 << B, false)));
        dp[0][0][grid[0][0]] = true;
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                for (int m = 0; m < (1 << B); ++m) {
                    if (r > 0) dp[r][c][m] = dp[r][c][m] || dp[r - 1][c][m ^ grid[r][c]];
                    if (c > 0) dp[r][c][m] = dp[r][c][m] || dp[r][c - 1][m ^ grid[r][c]];
                }
            }
        }
        for (int m = 0; m < (1 << B); ++m) {
            if (dp[R - 1][C - 1][m]) return m;
        }
        return -1;
    }
};
```

## Q4. Count Non Decreasing Arrays With Given Digit Sums

### Solution 1: dynamic programming, grouping, candidates, prefix sum

```cpp
const int MAXN = 5e3 + 1, MAXD = 51, MOD = 1e9 + 7;
class Solution {
private:
    int getDigitSum(int x) {
        int ans = 0;
        while (x > 0) {
            ans += x % 10;
            x /= 10;
        }
        return ans;
    }
public:
    int countArrays(vector<int>& digitSum) {
        int N = digitSum.size();
        vector<vector<int>> candidates(MAXD, vector<int>());
        for (int i = 0; i < MAXN; ++i) {
            int dsum = getDigitSum(i);
            candidates[dsum].emplace_back(i);
        }
        vector<int> dp(MAXN, 0), ndp(MAXN, 0), psum(MAXN, 1), npsum(MAXN, 0);
        dp[0] = 1;
        for (int d : digitSum) {
            ndp.assign(MAXN, 0);
            npsum.assign(MAXN, 0);
            for (int x : candidates[d]) {
                ndp[x] = psum[x];
            }
            for (int i = 0; i < MAXN; ++i) {
                npsum[i] = ndp[i];
                if (i > 0) npsum[i] += npsum[i - 1];
                npsum[i] %= MOD;
            }
            swap(ndp, dp);
            swap(npsum, psum);
        }
        int ans = accumulate(dp.begin(), dp.end(), 0, [](int accum, int x) { return (accum + x) % MOD; });
        return ans;
    }
};
```

# Leetcode Weekly Contest 495

## Q1. First Matching Character From Both Ends

### Solution 1: loop

For each index `i`, compare the character at `i` with its mirrored character at `N - 1 - i`.
The first place where they match is the answer, because the problem asks for the smallest such index.

```cpp
class Solution {
public:
    int firstMatchingIndex(string s) {
        int N = s.size();
        for (int i = 0; i < N; ++i) {
            if (s[i] == s[N - i - 1]) return i;
        }
        return -1;
    }
};
```

## Q2. Design Event Manager

### Solution 1: set, struct with custom comparator, map, online queries

Store the current priority of each event in a map, and also keep all active events in a set ordered by
highest priority first and then smallest event id. That makes both operations easy:
`updatePriority` removes the old pair and inserts the new one, and `pollHighest` just takes the first element of the set.

```cpp
struct Event {
    int event, priority;
    Event(int event, int priority) : event(event), priority(priority) {}
    bool operator<(const Event& other) const {
        if (priority != other.priority) return priority > other.priority;
        return event < other.event;
    }
};
class EventManager {
private:
    map<int, int> P;
    set<Event> S;
public:
    EventManager(vector<vector<int>>& events) {
        for (const auto &event : events) {
            int x = event[0], y = event[1];
            P[x] = y;
            S.emplace(x, y); // decreasing y, increasing x
        }
    }
    
    void updatePriority(int eventId, int newPriority) {
        int oldPriority = P[eventId];
        S.erase(Event(eventId, oldPriority));
        S.emplace(eventId, newPriority);
        P[eventId] = newPriority;
    }
    
    int pollHighest() {
        if (S.empty()) return -1;
        Event ev = *S.begin();
        S.erase(ev);
        return ev.event;
    }
};
```

## Q3. Sum of Sortable Integers

### Solution 1: finding divisors, array cycle, sorting

Only divisors of `N` can work, since the array is split into equal blocks of size `k`.
For a fixed `k`, each block must already be a cyclic shift of the block it should become in the fully sorted array.
So for every divisor, check each block independently, and add `k` to the answer if all blocks are feasible.

```cpp
class Solution {
private:
    bool feasible(int k, const vector<int> &nums, const vector<int> &target) {
        int N = nums.size();
        for (int i = 0; i < N; i += k) {
            // block from i to i + k - 1
            int pivot = i;
            for (int j = i + 1; j < i + k; ++j) {
                if (nums[j] < nums[j - 1]) {
                    if (pivot != i) return false;
                    pivot = j;
                }
            }
            vector<int> block;
            for (int j = pivot; j < i + k; ++j) {
                block.emplace_back(nums[j]);
            }
            for (int j = i; j < pivot; ++j) {
                block.emplace_back(nums[j]);
            }
            for (int j = i; j < i + k; ++j) {
                if (block[j - i] != target[j]) return false;
            }
        }
        return true;
    }
public:
    int sortableIntegers(vector<int>& nums) {
        int N = nums.size(), ans = 0;
        vector<int> divisors;
        for (int i = 1; 1LL * i * i <= N; ++i) {
            if (N % i == 0) {
                divisors.emplace_back(i);
                if (i != N / i) {
                    divisors.emplace_back(N / i);
                }
            }
        }
        vector<int> sortedNums = nums;
        sort(sortedNums.begin(), sortedNums.end());
        for (int d : divisors) {
            if (feasible(d, nums, sortedNums)) ans += d;
        }
        return ans;
    }
};
```

## Q4. Incremental Even-Weighted Cycle Queries

### Solution 1: union find, parity, undirected graph, xor

Use union find with an extra `parity` value that stores the xor/parity from a node to its component root.
If an edge connects two different components, merge them and set the new root relationship so the edge constraint is satisfied.
If the endpoints are already connected, then the edge is valid only when the xor between the two nodes matches the new edge weight.

```cpp
struct UnionFind {
    vector<int> parent, size, parity;
    UnionFind(int n) {
        parent.resize(n);
        iota(parent.begin(),parent.end(),0);
        size.assign(n,1);
        parity.assign(n, 0);
    }

    pair<int, int> find(int i) {
        if (i == parent[i]) {
            return {i, 0};
        }
        auto [p, x] = find(parent[i]);
        parent[i] = p;
        parity[i] ^= x;
        return {parent[i], parity[i]};
    }

    void unite(int i, int j, int w) {
        int ru, rv;
        tie(i, ru) = find(i);
        tie(j, rv) = find(j);
        if (i != j) {
            if (size[j] > size[i]) {
                swap(i, j);
            }
            size[i] += size[j];
            parent[j] = i;
            parity[j] ^= ru ^ rv ^ w;
        }
    }

    bool same(int i, int j) {
        return find(i).first == find(j).first;
    }
};
class Solution {
public:
    int numberOfEdgesAdded(int n, vector<vector<int>>& edges) {
        UnionFind dsu(n);
        int ans = 0;
        for (const auto& edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            if (dsu.same(u, v)) {
                if ((dsu.parity[u] ^ dsu.parity[v]) == w) ans++;
            } else {
                dsu.unite(u, v, w);
                ans++;
            }
        }
        return ans;
    }
};
```

# Leetcode Weekly Contest 496

## Q1. Mirror Frequency Distance

### Solution 1: frequency counting, mirrored pairs

Each letter or digit only interacts with its mirror partner, so the answer can be computed pair by pair.
For letters, the relevant pairs are `('a','z')`, `('b','y')`, ..., and for digits they are `('0','9')`, `('1','8')`, ....

If one side of a mirrored pair appears more often than the other, those extra copies are exactly the contribution of that pair, so we add `abs(freq[a] - freq[b])`.
After counting frequencies once, summing these independent contributions gives the total distance.

```cpp
class Solution {
public:
    int mirrorFrequency(string s) {
        map<char, int> freq;
        for (const char &ch : s) {
            freq[ch]++;
        }
        int ans = 0;
        for (int i = 0; i < 13; ++i) {
            char a = 'a' + i, b = 'a' + 25 - i;
            ans += abs(freq[a] - freq[b]);
        }
        for (int i = 0; i < 5; ++i) {
            char a = '0' + i, b = '0' + 9 - i;
            ans += abs(freq[a] - freq[b]);
        }
        return ans;
    }
};
```

## Q2. Integers With Multiple Sum of Two Cubes

### Solution 1: brute force over cube pairs, counting representations

Enumerate every unordered pair `(a, b)` with `1 <= a <= b` and check the value `a^3 + b^3`.
Using `a <= b` is the key detail because it counts each representation once instead of double-counting `(a, b)` and `(b, a)`.

`freq[x]` stores how many different cube pairs produce `x`.
At the end, every integer with `freq[x] >= 2` has at least two distinct representations as a sum of two positive cubes, so those are the values we return in sorted order.

```cpp
using int64 = long long;
class Solution {
public:
    vector<int> findGoodIntegers(int n) {
        unordered_map<int, int> freq;
        vector<int> ans;
        for (int b = 1; 1LL * b * b * b <= n; ++b) {
            for (int a = 1; a <= b; ++a) {
                int64 cand = 1LL * a * a * a + 1LL * b * b * b;
                if (cand > n) break;
                freq[cand]++;
            }
        }
        for (const auto &[k, v] : freq) {
            if (v < 2) continue;
            ans.emplace_back(k);
        }
        sort(ans.begin(), ans.end());
        return ans;
    }
};
```

## Q3. Minimum Increase to Maximize Special Indices

### Solution 1: alternating peaks, prefix/suffix cost

To maximize the number of special indices, the chosen positions must alternate, because two adjacent indices cannot both be strict peaks.
For any interior index `i`, the minimum cost to make it a peak is independent of the other non-adjacent choices:
raise `nums[i]` to `max(nums[i - 1], nums[i + 1]) + 1` if it is not already there.

`pref[i]` accumulates that cost for the odd indices up to `i`, which handles the only pattern when `N` is odd.
When `N` is even, the optimal maximum pattern can switch parity once, so we try every split: odd peaks on the left from `pref`, even peaks on the right from `suf`, and take the minimum total increase.

```cpp
using int64 = long long;
class Solution {
public:
    int64 minIncrease(vector<int>& nums) {
        int N = nums.size();
        vector<int64> pref(N, 0);
        for (int i = 1; i < N; ++i) {
            pref[i] = pref[i - 1];
            if (i % 2 == 0) continue;
            if (i + 1 >= N) continue;
            pref[i] += max(0, max(nums[i - 1], nums[i + 1]) + 1 - nums[i]);
        }
        int64 ans = pref.back();
        if (N & 1) return ans;
        int64 suf = 0;
        for (int i = N - 2; i >= 0; --i) {
            if (i & 1) continue;
            int64 cand = pref[i] + suf;
            ans = min(ans, cand);
            if (i > 0) suf += max(0, max(nums[i - 1], nums[i + 1]) + 1 - nums[i]);
        }
        return ans;
    }
};
```

## Q4. Minimum Operations to Achieve At Least K Peaks

### Solution 1: weighted independent set on a cycle, dynamic programming

Making index `i` a peak is cheapest if we only raise `nums[i]`, so its standalone cost is
`max(0, max(leftNeighbor, rightNeighbor) + 1 - nums[i])`.
After that, the real constraint is combinatorial: chosen peaks cannot be adjacent on the circular array.

That turns the problem into choosing `k` non-adjacent vertices on a cycle with minimum total weight.
The usual cycle trick is to split into two path cases: take index `0` and forbid its neighbors, or skip index `0`.
Inside `calc`, `dp[i][j]` is the minimum cost to process the first `i` positions of the path while choosing exactly `j` peaks, with transitions for skipping the current index or taking it from `i - 2`.

`cost[i]` stores the weight of choosing position `i`, and the final answer is the better of the two cycle-breaking cases.

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
private:
    vector<int> cost;
    int calc(int l, int r, int k, const vector<int>& nums) {
        if (l > r) return 0;
        if (k <= 0) return 0;
        int N = r - l + 1;
        vector<vector<int>> dp(N + 1, vector<int>(k + 1, INF));
        dp[0][0] = 0;
        if (k > 0) dp[1][1] = cost[l];
        for (int i = 1; i <= N; ++i) {
            for (int j = 0; j <= k; ++j) {
                // take this one as peak
                if (i > 1 && j > 0 && dp[i - 2][j - 1] != INF) {
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 1] + cost[l + i - 1]);
                }
                // skip this one as peak.
                dp[i][j] = min(dp[i][j], dp[i - 1][j]);
            }
        }
        return dp[N][k] < INF ? dp[N][k] : 0;
    }
public:
    int minOperations(vector<int>& nums, int k) {
        int N = nums.size();
        if (k > N / 2) return -1;
        cost.assign(N, 0);
        for (int i = 0; i < N; ++i) {
            cost[i] = max(0, max(nums[(i - 1 + N) % N], nums[(i + 1) % N]) + 1 - nums[i]);
        }
        int ans = min(cost[0] + calc(2, N - 2, k - 1, nums), calc(1, N - 1, k, nums));
        return ans;
    }
};
```

# Leetcode Biweekly Contest 180

## Q3. Minimum Operations to Transform Array into Alternating Prime

### Solution 1: sieve + greedy by position

The array positions are independent, because each operation only increments one element and the requirement is purely local: even indices must end up prime, and odd indices must end up non-prime. So for each index, the cheapest choice is simply the first valid number at or above the current value.

For even indices, we want the smallest prime >= nums[i]. After precomputing all primes with a sieve, we can binary search in the prime list P and jump directly to that next prime. The number of operations contributed by this position is exactly nextPrime - nums[i], and this is optimal because any smaller target is invalid and any larger target would cost more.

For odd indices, we want the smallest non-prime >= nums[i]. Since the valid target might be the current value already, the code just increments while nums[i] is prime. The moment it reaches a non-prime, that is the minimum possible number of increments for this position.

```cpp
const int MAXN = 1e5 + 5;
class Solution {
private:
    static bool precomputed;
    static bool primes[MAXN];
    static vector<int> P;
    void sieve(int n) {
        if (precomputed) return;
        precomputed = true;
        fill(primes, primes + n, true);
        primes[0] = primes[1] = false;
        int p = 2;
        for (int p = 2; p * p <= n; p++) {
            if (primes[p]) {
                for (int i = p * p; i < n; i += p) {
                    primes[i] = false;;
                }
            }
        }
        for (int i = 2; i < n; ++i) {
            if (primes[i]) {
                P.emplace_back(i);
            }
        }
    }
public:
    int minOperations(vector<int>& nums) {
        sieve(MAXN);
        int ans = 0, N = nums.size();
        for (int i = 0; i < N; ++i) {
            if (i % 2 == 0) {
                int j = lower_bound(P.begin(), P.end(), nums[i]) - P.begin();
                int cand = P[j] - nums[i];
                ans += cand;
            } else {
                while (primes[nums[i]]) {
                    ans++;
                    nums[i]++;
                }
            }
        }
        return ans;
    }
};

bool Solution::precomputed = false;
vector<int> Solution::P;
bool Solution::primes[MAXN];
```

## Q4. Maximum Value of Concatenated Binary Segments

### Solution 1: greedy sorting by concatenation order

Each pair (nums1[i], nums0[i]) defines a binary segment consisting of nums1[i] ones followed by nums0[i] zeros. The only freedom is how to order these segments before concatenating them into one big binary number. This is the same kind of greedy ordering used in problems like forming the largest concatenated number: for two segments A and B, we should decide their relative order by comparing A + B against B + A.

The key observation is that if placing A before B makes the combined binary string larger than placing B before A, then in an optimal arrangement A must appear before B. So sorting all segments with this pairwise comparator gives a globally optimal order. The comparator in the code does this without actually building the concatenated vectors: it compares the virtual sequences p1 + p2 and p2 + p1 element by element.

One subtle detail is the direction of the final value construction. The loop that computes ans processes the segments from least significant bit upward: within each segment it walks from right to left, and earlier segments in arr end up contributing lower bits. Because of that, the sort is written in the ascending version of the comparator, so the “better” segments are pushed later and become more significant in the final binary value.

```cpp
const int MOD = 1e9 + 7;
class Solution {
public:
    int maxValue(vector<int>& nums1, vector<int>& nums0) {
        int N = nums1.size();
        vector<vector<int>> arr;
        for (int i = 0; i < N; ++i) {
            vector<int> A;
            while (nums1[i]--) {
                A.emplace_back(1);
            }
            while (nums0[i]--) {
                A.emplace_back(0);
            }
            arr.emplace_back(A);
        }
        // sorted in ascending order
        sort(arr.begin(), arr.end(), [](const auto &p1, const auto &p2) {
            // propose the following that it is p1 + p2 > p2 + p1
            int n = p1.size(), m = p2.size();
            for (int i = 0; i < n + m; ++i) {
                int a = i < n ? p1[i] : p2[i - n];
                int b = i < m ? p2[i] : p1[i - m];
                if (a != b) return a < b;
            }
            return false;
        });
        int ans = 0, p = 1;
        for (const auto &vec : arr) {
            int M = vec.size();
            for (int i = M - 1; i >= 0; --i) {
                if (vec[i]) ans = (ans + p) % MOD;
                p = 2LL * p % MOD;
            }
        }
        return ans;
    }
};
```

# Leetcode Weekly Contest 497

## Q2. Angles of a Triangle

### Solution 1: geometry, math, implementation

Categories: triangle inequality, law of cosines, floating point math, sorting

Algorithm idea:
Sort the three side lengths so they are in increasing order. First check whether they can form a valid triangle using the triangle inequality a + b > c. If not, return an empty array. Otherwise, use the Law of Cosines to compute each of the three interior angles, convert from radians to degrees, then sort the angles before returning them.

```cpp
class Solution {
public:
    vector<double> internalAngles(vector<int>& sides) {
        sort(sides.begin(), sides.end());
        double a = sides[0], b = sides[1], c = sides[2];
        if (a + b <= c) return {};
        double pi = numbers::pi;
        double A = acos((b * b + c * c - a * a) / (2 * b * c)) * 180.0 / pi;
        double B = acos((a * a + c * c - b * b) / (2 * a * c)) * 180.0 / pi;
        double C = acos((a * a + b * b - c * c) / (2 * a * b)) * 180.0 / pi;
        vector<double> ans = {A, B, C};
        sort(ans.begin(), ans.end());
        return ans;
    }
};
```

## Q3. Longest Balanced Substring After One Swap

### Solution 1: strings, prefix sums, case analysis

Categories: prefix sum, hashmap or map, binary search, balanced substring, one modification

Algorithm idea:
Treat '0' as -1 and '1' as +1, so a balanced substring becomes a subarray whose sum is 0. The prefix sum lets you detect such substrings quickly. The map stores all earlier positions for each prefix value. Then the solution checks three cases while scanning from left to right:

a substring already balanced with no swap needed,
a substring that could be fixed by changing the balance by +2,
a substring that could be fixed by changing the balance by -2.

The stored first and last positions of 0 and 1, together with binary search on earlier prefix positions, are used to verify whether one swap can make that substring balanced and maximize its length.

```cpp
class Solution {
public:
    int longestBalanced(string s) {
        int N = s.size();
        int p0 = N, p1 = N, s0 = -1, s1 = -1;
        vector<int> pref(N, 0);
        for (int i = 0; i < N; ++i) {
            if (s[i] == '0') {
                if (p0 == N) p0 = i;
                s0 = i;
                pref[i] = -1;
            } else {
                if (p1 == N) p1 = i;
                s1 = i;
                pref[i] = 1;
            }
            if (i > 0) {
                pref[i] += pref[i - 1];
            }
        }
        int ans = 0;
        map<int, vector<int>> indexMap;
        indexMap[0].emplace_back(-1);
        for (int r = 0; r < N; ++r) {
            if (indexMap.find(pref[r]) != indexMap.end()) {
                int l = indexMap[pref[r]][0];
                ans = max(ans, r - l);
            }
            if (indexMap.find(pref[r] - 2) != indexMap.end()) {
                // pref[r] - pref[l] = 2, need 0
                const vector<int> &arr = indexMap[pref[r] - 2];
                if (s0 > r) {
                    int l = arr[0];
                    ans = max(ans, r - l);
                } else {
                    auto it = upper_bound(arr.begin(), arr.end(), p0);
                    if (it != arr.end()) {
                        int l = *it;
                        ans = max(ans, r - l);
                    }
                }
            }
            if (indexMap.find(pref[r] + 2) != indexMap.end()) {
                // need 1
                const vector<int> &arr = indexMap[pref[r] + 2];
                if (s1 > r) {
                    int l = arr[0];
                    ans = max(ans, r - l);
                } else {
                    auto it = upper_bound(arr.begin(), arr.end(), p1);
                    if (it != arr.end()) {
                        int l = *it;
                        ans = max(ans, r - l);
                    }
                }
            }
            indexMap[pref[r]].emplace_back(r);
        }
        return ans;
    }
};
```

## Q4. Good Subsequence Queries

### Solution 1: segment tree, range query, dynamic updates, number theory

Categories: segment tree, gcd, point updates, query processing, divisibility

Algorithm idea:
Build a segment tree where each node stores the gcd of its segment, but only values divisible by p are kept; otherwise the leaf stores 0. Since gcd(x, 0) = x, this lets the segment tree effectively track the gcd of only the relevant values. After each update query, update one position in the segment tree and inspect the overall gcd at the root.

If the root gcd is exactly p, then the current array may contain a good subsequence. For very small N, the code does an extra brute force check by temporarily removing each element one at a time to see whether some subsequence still gives gcd p. For larger N, it accepts the root gcd check directly.

When every number is divisible by p and the gcd of all numbers is p, could it happen that you are forced to use all of them?

What that situation would mean

If removing b_i breaks things, then all the other numbers must share some prime factor.

So for each index i, there must be a prime q_i such that:

q_i divides every number except b_i
q_i does not divide b_i

That means each element is the unique one “missing” some prime.

For 6 numbers, this is possible.

Example for p = 1:

[15015, 10010, 6006, 4290, 2730, 2310]

These are built from the 6 primes:

2, 3, 5, 7, 11, 13

Each number is missing exactly one of them.

So:

gcd of all 6 numbers is 1
but if you remove any one number, the remaining 5 all share the prime that was missing from that number
so every proper subset has gcd bigger than 1

So with 6 numbers, you really can be forced to take all of them.

Why this becomes impossible with more than 6 numbers

Suppose you had 7 such numbers.

Then you would need 7 distinct primes, say:

2, 3, 5, 7, 11, 13, 17

And each number would have to be divisible by all but one of those primes.

In particular, the number that is “missing 2” would have to be divisible by:

3⋅5⋅7⋅11⋅13⋅17=255255

But the problem says every number is at most 50000.

That is impossible.

So the bad pattern cannot exist for 7 numbers.

And if it cannot exist for 7, it also cannot exist for 8, 9, and so on.

The only way you are forced to take all elements is this special “each element is uniquely missing one prime” pattern.
That pattern can exist for at most 6 relevant numbers under the <= 50000 limit.

```cpp
int P;
template<class Node>
struct SegmentTree {
    struct Configuration {
        const Node neutral;                           // identity for merge
        function<Node(const Node&, const Node&)> merge;           // combine two nodes
    } config;

    int size = 0;
    vector<Node> nodes;

    SegmentTree(int n, Configuration config) : config(config) { init(n); }

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, config.neutral);
    }

    // this is for assign, for addition change to += val
    void update(int segment_idx, const Node& val) {
        segment_idx += size;
        if (val % P == 0) nodes[segment_idx] = val;
        else nodes[segment_idx] = 0;
        for (segment_idx >>= 1; segment_idx >= 1; segment_idx >>= 1) pull(segment_idx);
    }

    Node query(int left, int right) {
        left += size, right += size;
        Node left_acc = config.neutral;
        Node right_acc = config.neutral;
        while (left <= right) {
           if (left & 1) {
                // res on left
                left_acc = config.merge(left_acc, nodes[left++]);
            }
            if (~right & 1) {
                // res on right
                right_acc = config.merge(nodes[right--], right_acc);
            }
            left >>= 1, right >>= 1;
        }
        return config.merge(left_acc, right_acc);
    }
    private:
        inline void pull(int segment_idx) { nodes[segment_idx] = config.merge(nodes[segment_idx << 1], nodes[segment_idx << 1 | 1]); }
};

class Solution {
public:
    int countGoodSubseq(vector<int>& nums, int p, vector<vector<int>>& queries) {
        P = p;
        int N = nums.size(), M = queries.size();
        SegmentTree<int>::Configuration cfg{
            0,
            [](const int &x, const int &y) {
                return gcd(x, y);
            }
        };
        SegmentTree<int> seg(N, cfg);
        for (int i = 0; i < N; ++i) {
            seg.update(i, nums[i]);
        }
        int ans = 0;
        for (const auto &query : queries) {
            int idx = query[0], val = query[1];
            nums[idx] = val;
            seg.update(idx, val);
            if (N < 7 && seg.nodes[1] == p) {
                bool found = false;
                for (int i = 0; i < N; ++i) {
                    seg.update(i, 0);
                    if (seg.nodes[1] == p) {
                        found = true;
                    }
                    seg.update(i, nums[i]);
                }
                ans += found;
            } else if (seg.nodes[1] == p) {
                ans++;
            }
        }
        return ans;
    }
};
```

# Leetcode Weekly Contest 498

## Q2. Smallest Stable Index II

### Solution 1: prefix max, suffix min, linear scan

For each index i, the code wants to know:

the maximum value in nums[0..i]
the minimum value in nums[i..N-1]

So it precomputes smin[i] = min(nums[i..N-1]) using a suffix pass. Then it scans left to right while maintaining pmax = max(nums[0..i]). If at some index i we have:

pmax - smin[i] <= k

then i is stable, and since we scan from left to right, it is the smallest such index.

```cpp
class Solution {
public:
    int firstStableIndex(vector<int>& nums, int k) {
        int N = nums.size();
        vector<int> smin(N, 0);
        for (int i = N - 1; i >= 0; --i) {
            smin[i] = nums[i];
            if (i + 1 < N) smin[i] = min(smin[i], smin[i + 1]);
        }
        int pmax = 0;
        for (int i = 0; i < N; ++i) {
            pmax = max(pmax, nums[i]);
            if (pmax - smin[i] <= k) return i;
        }
        return -1;
    }
};
```

## Q3. Multi Source Flood Fill

### Solution 1: grid, multi-source bfs, sorting by color, queue

All source cells are inserted into the queue initially. That means the flood fill expands outward from every source at the same time. When an uncolored neighboring cell is first reached, it gets assigned the color of the source wave that reached it first.

The sources are sorted by descending color before being pushed into the queue. Since BFS processes equal-distance expansions in queue order, this gives priority to larger colors when multiple sources could reach a cell at the same distance.

```cpp
class Solution {
public:
    vector<vector<int>> colorGrid(int n, int m, vector<vector<int>>& sources) {
        vector<vector<int>> grid(n, vector<int>(m, 0));
        sort(sources.begin(), sources.end(), [](const auto &p1, const auto &p2) {
            return p1[2] > p2[2];
        });
        queue<pair<int, int>> q;
        for (const auto &p : sources) {
            int r = p[0], c = p[1], color = p[2];
            grid[r][c] = color;
            q.emplace(r, c);
        }
        while (!q.empty()) {
            auto [r, c] = q.front();
            q.pop();
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (abs(dr) + abs(dc) != 1) continue;
                    int nr = r + dr, nc = c + dc;
                    if (nr < 0 || nr >= n || nc < 0 || nc >= m) continue;
                    if (grid[nr][nc]) continue;
                    q.emplace(nr, nc);
                    grid[nr][nc] = grid[r][c];
                }
            }
        }
        return grid;
    }
};
```

## Q4. Count Good Integers on a Grid Path

### Solution 1: digit dp

The problem counts how many integers in the range [l, r] satisfy a digit constraint determined by a path on a grid.

The number is treated as a 16-digit zero-padded string. The path determines certain digit positions that must satisfy an ordering relationship. The vector indices stores the important positions along that path.

Then the code uses digit DP:

idx = current digit position
tight = whether we are still forced by the upper bound
other DP states track previous/path-related digit information needed to enforce the constraint

The function dfs(...) counts how many valid numbers can be formed from the current state up to the bound string.

As usual in digit DP, the final range answer is:

count(<= r) - count(<= l-1)

```cpp
using int64 = long long;
class Solution {
private:
    int64 dp[16][10][10][2][2];
    int64 dfs(const vector<int>& indices, const string& num, int idx, int lastDig, int prv, int tight, int zero) {
        int N = num.size();
        if (idx == N) {
            return 1;
        }
        if (dp[idx][lastDig][prv][tight][zero] != -1) return dp[idx][lastDig][prv][tight][zero];
        int64 ans = 0;
        bool found = find(indices.begin(), indices.end(), idx) != indices.end();
        for (int dig = found ? prv : 0; dig < 10; ++dig) {
            if (tight && dig > num[idx] - '0') break;
            ans += dfs(indices, num, idx + 1, dig, found ? dig : prv, dig == num[idx] - '0' && tight, dig == 0 && zero);
        }
        return dp[idx][lastDig][prv][tight][zero] = ans;
    }
public:
    int64 countGoodIntegersOnPath(int64 l, int64 r, string directions) {
        vector<int> indices;
        for (int i = 0, j = 0; i < 6; ++i) {
            indices.emplace_back(j);
            if (directions[i] == 'D') j += 4;
            else j++;
        }
        indices.emplace_back(15);
        string small = to_string(l - 1), large = to_string(r);
        small = string(max(0, 16 - static_cast<int>(small.size())), '0') + small;
        large = string(max(0, 16 - static_cast<int>(large.size())), '0') + large;
        fill(&dp[0][0][0][0][0], &dp[0][0][0][0][0] + 16 * 10 * 10 * 2 * 2, -1);
        int64 lb = dfs(indices, small, 0, 0, 0, 1, 1);
        fill(&dp[0][0][0][0][0], &dp[0][0][0][0][0] + 16 * 10 * 10 * 2 * 2, -1);
        int64 ub = dfs(indices, large, 0, 0, 0, 1, 1);
        return ub - lb;
    }
};
```

## 1559. Detect Cycles in 2D Grid

### Solution 1: dfs, grid, visited array, cycle detection

```cpp
class Solution {
private:
    vector<vector<bool>> vis;
    int R, C;
    vector<vector<char>> grid;
    bool inBounds(int r, int c) {
        return r >= 0 && r < R && c >= 0 && c < C;
    }
    bool dfs(int pr, int pc, int r, int c, int len = 1) {
        if (vis[r][c]) {
            return len >= 4;
        }
        vis[r][c] = true;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (abs(dr) + abs(dc) != 1) continue;
                int nr = r + dr, nc = c + dc;
                if (!inBounds(nr, nc)) continue;
                if (grid[nr][nc] != grid[r][c]) continue;
                if (nr == pr && nc == pc) continue;
                if (dfs(r, c, nr, nc, len + 1)) return true;
            }
        }
        return false;
    }
public:
    bool containsCycle(vector<vector<char>>& G) {
        grid = G;
        R = grid.size();C = grid[0].size();
        vis.assign(R, vector<bool>(C, false));
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                if (vis[r][c]) continue;
                if (dfs(r, c, r, c)) return true;
            }
        }
        return false;
    }
};
```

# Leetcode Weekly Contest 499

## Q1. Valid Elements in an Array

### Solution 1: prefix max, suffix max, linear scan

An element is valid if it is a new record high from either direction.

```cpp
class Solution {
public:
    vector<int> findValidElements(vector<int>& nums) {
        int N = nums.size();
        vector<bool> ans(N, false);
        int pmax = 0;
        for (int i = 0; i < N; ++i) {
            if (nums[i] > pmax) ans[i] = true;
            pmax = max(pmax, nums[i]);
        }
        int smax = 0;
        for (int i = N - 1; i >= 0; --i) {
            if (nums[i] > smax) ans[i] = true;
            smax = max(smax, nums[i]);
        }
        vector<int> res;
        for (int i = 0; i < N; ++i) {
            if (ans[i]) res.emplace_back(nums[i]);
        }
        return res;
    }
};
```

## Q3. Minimum Operations to Make Array Non Decreasing

### Solution 1: greedy

Every local drop contributes exactly the amount needed to fix it.

```cpp
using int64 = long long;
class Solution {
public:
    int64 minOperations(vector<int>& nums) {
        int N = nums.size();
        int64 ans = 0;
        for (int i = 1; i < N; ++i) {
            ans += max(0, nums[i - 1] - nums[i]);
        }
        return ans;
    }
};
```

## Q4. Maximum Sum of Alternating Subsequence With Distance at Least K

### Solution 1: dp, segment tree optimization, coordinate compression

Use DP for alternating subsequences, and use segment trees to quickly find the best previous compatible value that is at least k indices away.

```cpp
using int64 = long long;
const int64 INF = numeric_limits<int64>::max();
template<class Node>
struct SegmentTree {
    struct Configuration {
        const Node neutral;                           // identity for merge
        function<Node(const Node&, const Node&)> merge;           // combine two nodes
    } config;

    int size = 0;
    vector<Node> nodes;

    SegmentTree(int n, Configuration config) : config(config) { init(n); }

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, config.neutral);
    }

    // this is for assign, for addition change to += val
    void update(int segment_idx, const Node& val) {
        segment_idx += size;
        nodes[segment_idx] = val; // += val if want addition, to track frequency
        for (segment_idx >>= 1; segment_idx >= 1; segment_idx >>= 1) pull(segment_idx);
    }

    Node query(int left, int right) {
        left += size, right += size;
        Node left_acc = config.neutral;
        Node right_acc = config.neutral;
        while (left <= right) {
           if (left & 1) {
                // res on left
                left_acc = config.merge(left_acc, nodes[left++]);
            }
            if (~right & 1) {
                // res on right
                right_acc = config.merge(nodes[right--], right_acc);
            }
            left >>= 1, right >>= 1;
        }
        return config.merge(left_acc, right_acc);
    }
    private:
        inline void pull(int segment_idx) { nodes[segment_idx] = config.merge(nodes[segment_idx << 1], nodes[segment_idx << 1 | 1]); }
};

SegmentTree<int64>::Configuration cfg{
    -INF,
    [](const int64 &x, const int64 &y) {
        return max(x, y);
    }
};
class Solution {
public:
    int64 maxAlternatingSum(vector<int>& nums, int k) {
        int N = nums.size();
        vector<int64> up(N, 0), down(N, 0);
        vector<int> vals(nums.begin(), nums.end());
        sort(vals.begin(), vals.end());
        vals.erase(unique(vals.begin(), vals.end()), vals.end());
        int M = vals.size();
        vector<int> comp(N, 0);
        for (int i = 0; i < N; ++i) {
            int idx = lower_bound(vals.begin(), vals.end(), nums[i]) - vals.begin();
            comp[i] = idx;
        }
        SegmentTree<int64> segDown(N, cfg), segUp(N, cfg);
        for (int i = 0; i < N; ++i) {
            if (i - k >= 0) {
                // let's add it into the segment tree now. 
                int j = i - k;
                segDown.update(comp[j], max(1LL * nums[j], down[j]));
                segUp.update(comp[j], max(1LL * nums[j], up[j]));
            }
            // transitions
            // up comes from down, ans smaller value
            up[i] = max(1LL * nums[i], segDown.query(0, comp[i] - 1) + nums[i]);
            down[i] = max(1LL * nums[i], segUp.query(comp[i] + 1, M - 1) + nums[i]);
        }
        int64 ans = max(*max_element(up.begin(), up.end()), *max_element(down.begin(), down.end()));
        return ans;
    }
};
```

# Leetcode Weekly Contest 500

## Q2. Sum of Primes Between Number and Its Reverse

### Solution 1: prime in square root, string, reverse string

This solution reverses the number by converting it to a string, reversing the string, and converting it back to an integer.

```cpp
class Solution {
private:
    bool isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        int limit = static_cast<int>(sqrt(n));
        for (int i = 5; i <= limit; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }
public:
    int sumOfPrimesInRange(int n) {
        string s = to_string(n);
        reverse(s.begin(), s.end());
        int r = stoi(s);
        int ans = 0;
        for (int v = min(n, r); v <= max(n, r); ++v) {
            if (isPrime(v)) ans += v;
        }
        return ans;
    }
};
```

## Q3. Minimum Cost to Move Between Indices

### Solution 1: prefix sums, suffix sums, greedy, implementation

The direct cost across a range is the sum of adjacent gaps. Whenever the closest-index move goes in the needed direction, replace that edge cost with 1.

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    vector<int> minCost(vector<int>& nums, vector<vector<int>>& queries) {
        int N = nums.size(), M = queries.size();
        vector<int> C(N);
        for (int i = 0; i < N; ++i) {
            int best = INF;
            if (i > 0 && nums[i] - nums[i - 1] < best) {
                best = nums[i] - nums[i - 1];
                C[i] = i - 1;
            }
            if (i + 1 < N && nums[i + 1] - nums[i] < best) {
                best = nums[i + 1] - nums[i];
                C[i] = i + 1;
            }
        }
        vector<int> pref(N + 1, 0), suf(N + 1, 0);
        for (int i = 0; i < N; ++i) {
            if (i + 1 < N && C[i] == i + 1) {
                pref[i + 1] = nums[i + 1] - nums[i] - 1;
            }
            pref[i + 1] += pref[i];
        }
        for (int i = N - 1; i >= 0; --i) {
            if (i > 0 && C[i] == i - 1) {
                suf[i] = nums[i] - nums[i - 1] - 1;
            }
            suf[i] += suf[i + 1];
        }
        vector<int> ans(M);
        for (int i = 0; i < M; ++i) {
            int l = queries[i][0], r = queries[i][1];
            if (r >= l) {
                int delta = nums[r] - nums[l] - pref[r] + pref[l];
                ans[i] = delta;
            } else {
                int delta = nums[l] - nums[r] - suf[r + 1] + suf[l + 1];
                ans[i] = delta;
            }
        }
        return ans;
    }
};
```

## Q4. Maximize Fixed Points After Deletions

### Solution 1: longest increasing subsequence, greedy, sorting

Convert each usable element into a candidate fixed point, sort by deletion requirement, and find the longest compatible increasing sequence.

```cpp
class Solution {
public:
    int maxFixedPoints(vector<int>& nums) {
        int N = nums.size();
        vector<pair<int, int>> arr;
        for (int i = 0; i < N; ++i) {
            if (nums[i] > i) continue;
            arr.emplace_back(i - nums[i], nums[i]);
        }
        sort(arr.begin(), arr.end());
        vector<int> lis;
        for (const auto &[_, v] : arr) {
            int j = lower_bound(lis.begin(), lis.end(), v) - lis.begin();
            if (j == lis.size()) lis.emplace_back(v);
            lis[j] = v;
        }
        return lis.size();
    }
};
```

# Leetcode Biweekly Contest 182

## Q2. Minimum Flips to Make Binary String Coherent

### Solution 1: greedy, case analysis

The solution counts the number of 0s and 1s in the string, then compares a few possible coherent forms. Instead of trying every flip pattern, it uses the structure of the target string to derive the minimum number of flips from simple counts.

The key idea is that the answer can be found by considering limited cases, such as keeping mostly 1s with at most one exception, flipping all 0s, or handling edge 1s separately. This makes the solution linear in the length of the string.

```cpp
class Solution {
public:
    int minFlips(string s) {
        int cnt0 = 0, cnt1 = 0, N = s.size();
        for (char c : s) {
            if (c == '0') cnt0++;
            else cnt1++;
        }
        int ans = min(max(0, cnt1 - 1), cnt0);
        int cand = cnt1;
        if (s[0] == '1') cand--;
        if (N > 1 && s.back() == '1') cand--;
        ans = min(ans, cand);
        return ans;
    }
};
```

## Q3. Minimum Generations to Target Point

### Solution 1: simulation, bfs generation

The solution starts with the given set of 3D points. In each generation, it considers every pair of existing points and creates a new point using the coordinate-wise average:

x = (x1 + x2) / 2
y = (y1 + y2) / 2
z = (z1 + z2) / 2

Each newly generated point is added only if it has not been seen before. After every generation, the algorithm checks whether the target point has appeared.

This behaves like breadth-first expansion over possible generated points. Since the coordinates are bounded by the problem constraints, the number of possible states is finite.

```cpp
class Solution {
public:
    int minGenerations(vector<vector<int>>& points, vector<int>& target) {
        set<vector<int>> seen;
        for (const auto &point : points) {
            seen.emplace(point);
        }
        for (int k = 0; ; k++) {
            if (seen.find(target) != seen.end()) {
                return k;
            }
            vector<vector<int>> nxt;
            int N = points.size();
            for (int i = 0; i < N; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    int x1 = points[i][0], y1 = points[i][1], z1 = points[i][2];
                    int x2 = points[j][0], y2 = points[j][1], z2 = points[j][2];
                    int x = (x1 + x2) / 2, y = (y1 + y2) / 2, z = (z1 + z2) / 2;
                    vector<int> point = {x, y, z};
                    if (seen.find(point) == seen.end()) {
                        nxt.emplace_back(point);
                        seen.emplace(point);
                    }
                }
            }
            if (nxt.empty()) break;
            points.insert(points.end(), nxt.begin(), nxt.end());
        }
        return -1;
    }
};
```

## Q4. Minimum Threshold Path With Limited Heavy Edges

### Solution 1: binary search, 0-1 bfs, shortest path, undirected graph

The goal is to find the minimum threshold value such that there exists a path from source to target using at most k heavy edges.

For a guessed threshold target, every edge is treated as either:

0 cost if w <= target
1 cost if w > target

Then the algorithm runs 0-1 BFS to find the minimum number of heavy edges needed to reach the destination.

Because the condition is monotonic, binary search is used:

If a threshold works, try a smaller one.
If it does not work, try a larger one.

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
private:
    int N;
    vector<vector<pair<int, int>>> adj;
    int bfs(int s, int t, int target) {
        vector<int> dp(N, INF);
        deque<int> dq;
        dq.emplace_back(s);
        dp[s] = 0;
        while (!dq.empty()) {
            int u = dq.front();
            dq.pop_front();
            if (u == t) {
                return dp[t];
            }
            for (const auto &[v, w] : adj[u]) {
                if (w > target && dp[u] + 1 < dp[v]) {
                    dq.emplace_back(v);
                    dp[v] = dp[u] + 1;
                } else if (w <= target && dp[u] < dp[v]) {
                    dq.emplace_front(v);
                    dp[v] = dp[u];
                }
            }
        }
        return dp[t];
    }
public:
    int minimumThreshold(int n, vector<vector<int>>& edges, int source, int target, int k) {
        adj.assign(n, vector<pair<int, int>>());
        N = n;
        for (const auto &edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            adj[u].emplace_back(v, w);
            adj[v].emplace_back(u, w);
        }
        int lo = 0, hi = 1e9 + 5;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (bfs(source, target, mid) <= k) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return bfs(source, target, lo) <= k ? lo : -1;
    }
};
```

# Leetcode Weekly Contest 501

## Q2. Count Valid Word Occurrences

### Solution 1: string parsing, map counting

The solution first concatenates all chunks into one string. Then it scans the string and extracts valid words according to the problem's rules.

A word may include lowercase letters and valid hyphens, but a hyphen only belongs to a word if it is surrounded by letters. Invalid separators, whitespace, and invalid hyphen patterns terminate the current word.

Each extracted word is counted in a map. Then each query is answered by looking up the word frequency.

```cpp
class Solution {
public:
    vector<int> countWordOccurrences(vector<string>& chunks, vector<string>& queries) {
        string S;
        for (const string& chunk : chunks) {
            S += chunk;
        }
        int N = S.size();
        map<string, int> words;
        for (int i = 0; i < N; ) {
            while (i < N) {
                if (!isalpha(S[i])) {
                    i++;
                } else {
                    break;
                }
            }
            int start = i;
            while (i < N) {
                if (isspace(S[i])) {
                    break;
                } else if (i + 1 < N && S[i] == '-' && !isalpha(S[i + 1])) {
                    break;
                } else if (i + 1 == N && S[i] == '-') {
                    break;
                }
                i++;
            }
            string cand = S.substr(start, i - start);
            words[cand]++;
        }
        vector<int> ans;
        for (const string &sq : queries) {
            ans.emplace_back(words[sq]);
        }
        return ans;
    }
};
```

## Q3. Minimize Array Sum Using Divisible Replacements

### Solution 1: greedy, number theory, counting frequencies, harmonic series

The solution counts how many times each number appears. Then it iterates from small numbers to large numbers.

For every number i, it absorbs the frequency of all multiples of i. This represents replacing divisible larger values with a smaller divisor whenever possible. Since smaller values reduce the total sum, processing numbers in increasing order greedily gives the minimized sum.

```cpp
using int64 = long long;
class Solution {
public:
    int64 minArraySum(vector<int>& nums) {
        unordered_map<int, int> freq;
        for (int x : nums) {
            freq[x]++;
        }
        int N = *max_element(nums.begin(), nums.end());
        for (int i = 1; i < N; ++i) {
            if (!freq[i]) continue;
            for (int j = 2 * i; j <= N; j += i) {
                if (!freq[j]) continue;
                freq[i] += freq[j];
                freq[j] = 0;
            }
        }
        int64 ans = 0;
        for (const auto &[x, y] : freq) {
            ans += 1LL * x * y;
        }
        return ans;
    }
};
```

## Q4. Minimum Cost to Buy Apples II

### Solution 1: all pairs shortest paths, undirected graph, dijkstra

For each city, the goal is to either buy an apple there or travel to another city, buy an apple, and pay both travel cost and tax cost.

The solution builds two graphs:

A graph weighted by normal travel cost.
A graph weighted by tax-adjusted travel cost.

It then runs Dijkstra from every city on both graphs to compute all-pairs shortest paths.

For every starting city i, it tries every possible buying city j:

candidate = prices[j] + costDist[i][j] + taxDist[i][j]

The minimum candidate becomes the answer for city i.

```cpp
using int64 = long long;
template <class Weight>
constexpr Weight INF_VALUE()
{
    return numeric_limits<Weight>::max();
}

template <class Weight>
vector<Weight> dijkstra(int n, int src, const vector<vector<pair<int, Weight>>> &adj)
{
    const Weight INF = INF_VALUE<Weight>();
    vector<Weight> dist(n, INF);
    dist[src] = 0;
    priority_queue<pair<Weight, int>, vector<pair<Weight, int>>, greater<pair<Weight, int>>> minheap;
    minheap.emplace(0, src);

    while (!minheap.empty())
    {
        auto [d, u] = minheap.top();
        minheap.pop();

        if (d != dist[u])
            continue;

        for (const auto &[v, w] : adj[u])
        {
            if (dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                minheap.emplace(dist[v], v);
            }
        }
    }
    return dist;
}

template <class Weight>
vector<vector<Weight>> shortestPaths(int n, const vector<array<Weight, 3>> &edges)
{
    vector<vector<pair<int, Weight>>> adj(n);
    for (const auto &e : edges)
    {
        int u = e[0], v = e[1];
        Weight w = e[2];
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }
    vector<vector<Weight>> dist(n);
    for (int src = 0; src < n; src++)
    {
        dist[src] = dijkstra<Weight>(n, src, adj);
    }
    return dist;
}

class Solution
{
public:
    vector<int> minCost(int n, vector<int> &prices, vector<vector<int>> &roads)
    {
        int M = roads.size();
        vector<array<int64, 3>> costEdges(M), taxEdges(M);
        for (int i = 0; i < M; ++i)
        {
            int u = roads[i][0], v = roads[i][1], c = roads[i][2], t = roads[i][3];
            costEdges[i] = {u, v, c};
            taxEdges[i] = {u, v, 1LL * c * t};
        }
        vector<vector<int64>> costDist = shortestPaths<int64>(n, costEdges);
        vector<vector<int64>> taxDist = shortestPaths<int64>(n, taxEdges);
        vector<int> ans(prices.begin(), prices.end());
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (i == j)
                    continue;
                if (costDist[i][j] == INF_VALUE<int64>() || taxDist[i][j] == INF_VALUE<int64>())
                    continue;
                int64 cand = prices[j] + costDist[i][j] + taxDist[i][j];
                ans[i] = min<int64>(ans[i], cand);
            }
        }
        return ans;
    }
};
```

## 1665. Minimum Initial Energy to Finish Tasks

### Solution 1: greedy, scheduling optimization

Sort by actual - minimum decreasing, then build the required starting energy backward using:

```cpp
class Solution {
public:
    int minimumEffort(vector<vector<int>>& tasks) {
        sort(tasks.begin(), tasks.end(), [](const auto &a, const auto &b) {
            return a[1] - a[0] < b[1] - b[0];
        });
        int ans = 0;
        for (const auto &task : tasks) {
            ans = max(ans + task[0], task[1]);
        }
        return ans;
    }
};
```

# Leetcode Weekly Contest 502

## Q2. Count K-th Roots in a Range

### Solution 1: numnber theory, brute force, implementation

```cpp
using int64 = long long;
class Solution {
public:
    int countKthRoots(int l, int r, int k) {
        if (k == 1) return r - l + 1;
        int ans = 0;
        for (int i = 0; 1LL * i * i <= r; ++i) {
            int64 cand = 1;
            for (int j = 0; j < k; ++j) {
                cand *= i;
                if (cand > r) break;
            }
            if (cand < l || cand > r) continue;
            ans++;
        }
        return ans;
    }
};
```

## Q3. Largest Local Values in a Matrix II

### Solution 1: grid, 2d prefix sum, value enumeration

So the main idea is:

Fix a possible value x.
Build a prefix sum of cells greater than x.
For every cell equal to x, quickly ask: “Is there anything bigger nearby?”
If no, count it.

```cpp
class Solution {
private:
    int query(const vector<vector<int>> &ps, int r1, int c1, int r2, int c2) {
        if (r1 > r2 || c1 > c2) return 0;
        return ps[r2 + 1][c2 + 1] - ps[r1][c2 + 1] - ps[r2 + 1][c1] + ps[r1][c1];
    }
public:
    int countLocalMaximums(vector<vector<int>>& matrix) {
        int R = matrix.size(), C = matrix[0].size();
        int ans = 0;
        for (int x = 1; x <= 200; ++x) {
            vector<vector<int>> ps(R + 1, vector<int>(C + 1, 0));
            for (int r = 1; r <= R; r++) {
                for (int c = 1; c <= C; c++) {
                    int val = int(matrix[r - 1][c - 1] > x);
                    ps[r][c] = ps[r - 1][c] + ps[r][c - 1] - ps[r - 1][c - 1] + val;
                }
            }
             for (int r = 0; r < R; ++r) {
                for (int c = 0; c < C; ++c) {
                    if (matrix[r][c] != x) continue;
                    int r1 = max(0, r - x);
                    int r2 = min(R - 1, r + x);
                    int c1 = max(0, c - x);
                    int c2 = min(C - 1, c + x);
                    int cnt = query(ps, r1, c1, r2, c2);
                    // need to remove the corners
                    vector<pair<int, int>> corners = {
                        {r - x, c - x},
                        {r - x, c + x},
                        {r + x, c - x},
                        {r + x, c + x}
                    };
                    for (const auto &[nr, nc] : corners) {
                        if (nr >= 0 && nr < R && nc >= 0 && nc < C && matrix[nr][nc] > x) {
                            cnt--;
                        }
                    }
                    if (cnt == 0) ans++;
                }
            }
        }
        return ans;
    }
};
```

## Q4. Smallest Unique Subarray

### Solution 1: suffix array, longest common prefix

You want the shortest subarray that appears exactly once. A subarray is the same as a prefix of some suffix. So the problem becomes:
- For every suffix of the array, what is the shortest prefix of that suffix that is different from every other suffix?
That is exactly what suffix arrays and LCP arrays are good at.

Suffix arrays are powerful here because they transform a hard global duplicate-subarray problem into a local neighbor comparison problem.

Without suffix array, checking whether every subarray appears elsewhere can be expensive.

With suffix array:

1. Every suffix is sorted.
2. Similar suffixes become adjacent.
3. The LCP array tells us how much overlap adjacent suffixes have.
4. For each suffix, the shortest unique prefix is determined only by neighboring LCP values.

So the core formula is:
shortest unique prefix length=1+max(LCP with previous suffix,LCP with next suffix)
Then take the minimum over all suffixes.

How short of a prefix can I take so that no other suffix has the same prefix?
For suffix sa[i], its biggest conflict with another suffix comes from its closest sorted neighbors:

so in other words you find for each suffix the longest possible prefix that exists elsewhere in the sequence, then just add one more to that to get the smallest possible prefix that appears exactly once. and minimize over all the suffixes considered.

smallest unique subarray: find the smallest prefix that escapes appearing elsewhere

```cpp
class Solution {
private:
    template <typename Seq>
    vector<int> lcp_construction(const Seq& s, const vector<int>& p) {
        int n = s.size();
        vector<int> rank(n, 0);
        for (int i = 0; i < n; i++)
            rank[p[i]] = i;
    
        int k = 0;
        vector<int> lcp(n-1, 0);
        for (int i = 0; i < n; i++) {
            if (rank[i] == n - 1) {
                k = 0;
                continue;
            }
            int j = p[rank[i] + 1];
            while (i + k < n && j + k < n && s[i+k] == s[j+k])
                k++;
            lcp[rank[i]] = k;
            if (k)
                k--;
        }
        return lcp;
    }
    
    void radix_sort(const vector<int>& equivalence_class, vector<int>& leaderboard, vector<int>& update_leaderboard) {
        int n = leaderboard.size();
        vector<int> bucket_size(n, 0), bucket_pos(n, 0);
        bucket_size.assign(n, 0);
        for (int eq_class : equivalence_class) {
            bucket_size[eq_class]++;
        }
        bucket_pos.assign(n, 0);
        for (int i = 1; i < n; i++) {
            bucket_pos[i] = bucket_pos[i - 1] + bucket_size[i - 1];
        }
        update_leaderboard.assign(n, 0);
        for (int i = 0; i < n; i++) {
            int eq_class = equivalence_class[leaderboard[i]];
            int pos = bucket_pos[eq_class];
            update_leaderboard[pos] = leaderboard[i];
            bucket_pos[eq_class]++;
        }
    }
    template <typename Seq>
    vector<int> suffix_array(const Seq& s) {
        int n = s.size();
        using T = typename Seq::value_type;
        vector<pair<T, int>> arr(n);
        for (int i = 0; i < n; i++) {
            arr[i] = {s[i], i};
        }
        sort(arr.begin(), arr.end());
        vector<int> leaderboard(n, 0), equivalence_class(n, 0);
        for (int i = 0; i < n; i++) {
            leaderboard[i] = arr[i].second;
        }
        equivalence_class[leaderboard[0]] = 0;
        for (int i = 1; i < n; i++) {
            T left_segment = arr[i - 1].first;
            T right_segment = arr[i].first;
            equivalence_class[leaderboard[i]] = equivalence_class[leaderboard[i - 1]] + (left_segment != right_segment);
        }
        int k = 1;
        vector<int> update_equivalence_class(n, 0), update_leaderboard(n, 0);
        while (k < n) {
            for (int i = 0; i < n; i++) {
                leaderboard[i] = (leaderboard[i] - k + n) % n; // create left segment, keeps sort of the right segment
            }
            radix_sort(equivalence_class, leaderboard, update_leaderboard); // radix sort for the left segment
            swap(leaderboard, update_leaderboard);
            update_equivalence_class.assign(n, 0);
            update_equivalence_class[leaderboard[0]] = 0;
            for (int i = 1; i < n; i++) {
                pair<int, int> left_segment = {equivalence_class[leaderboard[i - 1]], equivalence_class[(leaderboard[i - 1] + k) % n]};
                pair<int, int> right_segment = {equivalence_class[leaderboard[i]], equivalence_class[(leaderboard[i] + k) % n]};
                update_equivalence_class[leaderboard[i]] = update_equivalence_class[leaderboard[i - 1]] + (left_segment != right_segment);
            }
            k <<= 1;
            swap(equivalence_class, update_equivalence_class);
        }
        return leaderboard;
    }
public:
    int smallestUniqueSubarray(vector<int>& nums) {
        int N = nums.size();
        nums.emplace_back(0); // sentinel node
        vector<int> sa = suffix_array(nums);
        vector<int> lcp = lcp_construction(nums, sa);
        int ans = N;
        for (int i = 0; i <= N; ++i) {
            int idx = sa[i];
            if (idx == N) continue; // sentinel node
            int prefixMax = 0;
            if (i > 0) prefixMax = max(prefixMax, lcp[i - 1]);
            if (i < N) prefixMax = max(prefixMax, lcp[i]);
            int cand = prefixMax + 1;
            if (cand > N - idx) continue;
            ans = min(ans, cand);
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 183

## Q2. Minimum Operations to Make Array Modulo Alternating I

### Solution 1: modular arithmetic, greedy

Try every alternating modulo pattern and choose the cheapest one.

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
private:
    int cost(int k, int val, int x) {
        int cand = INF;
        // increase
        if (val <= x) cand = min(cand, x - val);
        else cand = min(cand, k - (val - x));
        // decrease
        if (val >= x) cand = min(cand, val - x);
        else cand = min(cand, k - (x - val));
        return cand;
    }
public:
    int minOperations(vector<int>& nums, int k) {
        int N = nums.size();
        int ans = INF;
        for (int x = 0; x < k; ++x) {
            for (int y = 0; y < k; ++y) {
                if (x == y) continue;
                int cur = 0;
                for (int i = 0; i < N; ++i) {
                    int val = nums[i] % k;
                    if (i % 2 == 0) {
                        cur += cost(k, val, x);
                    } else {
                        cur += cost(k, val, y);
                    }
                }
                ans = min(ans, cur);
            }
        }
        return ans;
    }
};
```

## Q3. Maximum Path Intersection Sum in a Grid

### Solution 1: prefix sums, grid, kadane variation

The algorithm checks three possible kinds of valid intersections or paths:

Horizontal segments
For each row, it scans left to right using prefix sums.
It keeps the smallest previous prefix sum seen so far.
The best subarray ending at the current column is:
currentPrefix - minimumPreviousPrefix
Vertical segments
It does the same thing for each column, scanning top to bottom.
This finds the maximum vertical contiguous segment.
Single interior cells
It separately considers cells that are not on the border.
This handles cases where the best “intersection” is just one internal cell.

Core idea:
Reduce each row and column to a maximum subarray problem using prefix sums.

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int maxScore(vector<vector<int>>& grid) {
        int R = grid.size(), C = grid[0].size();
        int ans = -INF;
        // think about it, length 1 is impossible? 
        for (int r = 0; r < R; ++r) {
            int psum = 0, minSum = INF;
            for (int c = 0; c < C; ++c) {
                int ppsum = psum;
                psum += grid[r][c];
                if (c > 0) ans = max(ans, psum - minSum);
                minSum = min(minSum, ppsum);
            }
        }
        for (int c = 0; c < C; ++c) {
            int psum = 0, minSum = INF;
            for (int r = 0; r < R; ++r) {
                int ppsum = psum;
                psum += grid[r][c];
                if (r > 0) ans = max(ans, psum - minSum);
                minSum = min(minSum, ppsum);
            }
        }
        for (int r = 1; r + 1 < R; ++r) {
            for (int c = 1; c + 1 < C; ++c) {
                ans = max(ans, grid[r][c]);
            }
        }
        return ans;
    }
};
```

## Q4. Count Non Adjacent Subsets in a Rooted Tree

### Solution 1: tree dp, postorder dfs, combinatorics, knapsack like merging, modular sum counting, convolution

The problem counts subsets of tree nodes such that:

No two selected nodes are adjacent.
The sum of selected values is divisible by k.

The solution uses two DP states for every node u:

dp0[u][r] = number of valid subsets in subtree u
            where u is NOT selected
            and subset sum % k == r

dp1[u][r] = number of valid subsets in subtree u
            where u IS selected
            and subset sum % k == r

Initialization:

dp0[u][0] = 1
dp1[u][nums[u] % k] = 1

Then during DFS, each child subtree is merged into the current node.

If u is selected, then its child v cannot be selected:

dp1[u] combines only with dp0[v]

If u is not selected, then child v may either be selected or not selected:

dp0[u] combines with dp0[v] + dp1[v]

The modulo remainder is updated like:

newRemainder = (i + j) % k

Finally, the answer is:

dp0[0][0] + dp1[0][0] - 1

The -1 removes the empty subset.

$$C[r] = \sum_{\substack{0 \le i < K \\ 0 \le j < K \\ (i + j) \bmod K = r}} A[i] \cdot B[j]$$

Convolution is a way to merge two sequences into a new sequence where each output position collects contributions from pairs of input positions.

In your DP, the two sequences are:

A[i] = number of ways from the already-processed part of u's subtree with remainder i
B[j] = number of ways from child v's subtree with remainder j

The merged sequence is:

C[r] = number of ways after combining them with remainder r

For each node u, build a remainder-count sequence for its subtree by starting with the choice at u, then sequentially convolving in each child subtree’s remainder-count sequence, while respecting whether the child root may be selected.

Why is it multiplication? 

Because each side represents a number of choices, and when two choices can be combined freely, you use the product rule.

multiply = combine independent compatible choices

They are dependent computationally, but independent combinatorially.

Computationally:

dp after v2 depends on dp after v1

because you are accumulating counts.

```cpp
const int MOD = 1e9 + 7;
class Solution {
private:
    int K, N;
    vector<int> ndp0, ndp1, arr;
    vector<vector<int>> dp0, dp1, adj;
    void dfs(int u, int p = -1) {
        dp0[u][0] = 1;
        dp1[u][arr[u]] = 1;
        for (int v : adj[u]) {
            if (v == p) continue;
            dfs(v, u);
            ndp0.assign(K, 0);
            ndp1.assign(K, 0);
            // take 1
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < K; ++j) {
                    int val = (i + j) % K;
                    ndp1[val] = (ndp1[val] + (1LL * dp1[u][i] * dp0[v][j]) % MOD) % MOD;
                    int cnt = (dp0[v][j] + dp1[v][j]) % MOD;
                    ndp0[val] = (ndp0[val] + (1LL * dp0[u][i] * cnt) % MOD) % MOD;
                }
            }
            swap(dp1[u], ndp1);
            swap(dp0[u], ndp0);
        }
    }
public:
    int countValidSubsets(vector<int>& parent, vector<int>& nums, int k) {
        N = parent.size();
        K = k;
        for (int &x : nums) {
            x %= k;
        }
        arr = nums;
        adj.assign(N, vector<int>());
        for (int i = 1; i < N; ++i) {
            adj[i].emplace_back(parent[i]);
            adj[parent[i]].emplace_back(i);
        }
        dp0.assign(N, vector<int>(K, 0));
        dp1.assign(N, vector<int>(K, 0));
        dfs(0);
        int ans = (dp0[0][0] + dp1[0][0]) % MOD;
        ans--;
        if (ans < 0) ans += MOD;
        return ans;
    }
};
```

# Leetcode Weekly Contest 503

## Q2. Password Strength

### Solution 1: string parsing, set, simulation

```cpp
class Solution {
public:
    int passwordStrength(string password) {
        int ans = 0;
        unordered_set<char> seen;
        for (char c : password) {
            if (seen.find(c) != seen.end()) continue;
            seen.emplace(c);
            if (c >= 'a' && c <= 'z') {
                ans++;                
            } else if (c >= 'A' && c <= 'Z') {
                ans += 2;
            } else if (c >= '0' && c <= '9') {
                ans += 3;
            } else {
                ans += 5;
            }
        }
        return ans;
    }
};
```

## Q3. Minimum Operations to Sort a Permutation

### Solution 1: greedy, case analysis, permuation

The goal is to determine whether the permutation can become sorted using rotations and optionally one reverse operation.

The helper function calc(arr) finds where 0 is located. Since the sorted permutation is:

0, 1, 2, 3, ...

after rotation, the array must look like sorted order starting from 0.

So calc checks whether:

arr[(i + pivot) % N] == i

for every i.

If true, then the array can be sorted by rotating left pivot times.

The main function checks several cases:

Case 1: No reverse

Check whether the original array is a rotated sorted array.

x = calc(nums)

If yes, answer can be x.

Case 2: Rotate the other direction

If the array can be sorted by left rotations, then it can also be viewed through right rotations, but the code adds extra operations based on the problem’s operation rules:

(N - x) % N + 2
Case 3: Reverse first

Reverse the array first, then check if the reversed array can be sorted by rotations.

reverse(nums1.begin(), nums1.end());
cand = calc(nums1);
ans = min(ans, cand + 1);

The +1 is for the reverse operation.

Case 4: Reverse last

If the reversed version is rotatable into sorted order, that also corresponds to doing rotations first and then reversing at the end:

(N - cand) % N + 1

Why this works:
A permutation that can be sorted with these operations must be either:

already a rotation of sorted order, or
a rotation of reversed sorted order.

So checking the original and reversed array covers the possible structures.

```cpp
const int INF = numeric_limits<int>::max();
class Solution {
private:
    int calc(const vector<int> &arr) {
        int N = arr.size();
        int pivot = 0;
        for (int i = 0; i < N; ++i) {
            if (arr[i] == 0) {
                pivot = i;
                break;
            }
        }
        // because permutation of 0,1,2,...,N
        for (int i = 0; i < N; ++i) {
            if (arr[(i + pivot) % N] != i) return INF;
        }
        return pivot;
    }
public:
    int minOperations(vector<int>& nums) {
        int ans = INF, N = nums.size();
        // case 1, no reverse
        int x = calc(nums);
        if (x < INF) {
             ans = min(ans, x);
            // case 2, right rotations
            ans = min(ans, (N - x) % N + 2);
        }
        // case 2, reverse first
        vector<int> nums1(nums.begin(), nums.end());
        reverse(nums1.begin(), nums1.end());
        int cand = calc(nums1);
        if (cand < INF) ans = min(ans, cand + 1);
        // case 3, reverse last
        if (cand < INF) {
            ans = min(ans, (N - cand) % N + 1);
        }
        return ans < INF ? ans : -1;
    }
};
```

## Q4. Number of Pairs After Increment

### Solution 1: square root decomposition, lazy propagation, frequency counting, map, range add update

Square root decomposition idea

Split nums2 into blocks of size about sqrt(n).

For each block, store:

unordered_map<int64, int> freq;

This map counts how many raw values exist inside that block.

Also store:

lazy[b]

which means every value in block b should be treated as having lazy[b] added to it.

Range update

For partial blocks, push the lazy value into the actual array, update elements directly, then rebuild the frequency map.

For full blocks, do not touch individual elements. Just do:

lazy[b] += val;
Query

To count values equal to target, each block checks:

rawTarget = target - lazy[b]

because the actual visible value is:

arr[i] + lazy[b]

So if we want:

arr[i] + lazy[b] == target

then:

arr[i] == target - lazy[b]

The frequency map gives that count quickly.

Why this works:
Lazy values allow full block range updates in O(1) per block, while frequency maps allow fast counting by value.

```cpp
using int64 = long long;
struct SqrtDecomp {
    int N, B, numBlocks;
    vector<int64> arr;
    vector<int64> lazy;
    vector<unordered_map<int64, int>> freq;

    SqrtDecomp(vector<int>& nums) {
        N = nums.size();
        B = sqrt(N) + 1;
        numBlocks = (N + B - 1) / B;

        arr.assign(nums.begin(), nums.end());
        lazy.assign(numBlocks, 0);
        freq.assign(numBlocks, unordered_map<int64, int>());

        build();
    }

    int block(int i) {
        return i / B;
    }

    int start(int b) {
        return b * B;
    }

    int end(int b) {
        return min(N - 1, (b + 1) * B - 1);
    }

    void build() {
        for (int b = 0; b < numBlocks; b++) {
            rebuild(b);
        }
    }

    void rebuild(int b) {
        freq[b].clear();

        for (int i = start(b); i <= end(b); i++) {
            freq[b][arr[i]]++;
        }
    }

    void push(int b) {
        if (lazy[b] == 0) return;

        for (int i = start(b); i <= end(b); i++) {
            arr[i] += lazy[b];
        }

        lazy[b] = 0;
        rebuild(b);
    }

    void updateRange(int l, int r, int val) {
        int bl = block(l);
        int br = block(r);

        if (bl == br) {
            push(bl);

            for (int i = l; i <= r; i++) {
                arr[i] += val;
            }

            rebuild(bl);
            return;
        }

        // Left partial block
        push(bl);
        for (int i = l; i <= end(bl); i++) {
            arr[i] += val;
        }
        rebuild(bl);

        // Full middle blocks
        for (int b = bl + 1; b <= br - 1; b++) {
            lazy[b] += val;
        }

        // Right partial block
        push(br);
        for (int i = start(br); i <= r; i++) {
            arr[i] += val;
        }
        rebuild(br);
    }

    int query(int64 target) {
        int ans = 0;

        for (int b = 0; b < numBlocks; b++) {
            int64 rawTarget = target - lazy[b];

            auto it = freq[b].find(rawTarget);
            if (it != freq[b].end()) {
                ans += it->second;
            }
        }

        return ans;
    }
};
class Solution {
private:
public:
    vector<int> numberOfPairs(vector<int>& nums1, vector<int>& nums2, vector<vector<int>>& queries) {
        SqrtDecomp A(nums2);
        vector<int> ans;
        for (const auto &query : queries) {
            int t = query[0];
            if (t == 1) {
                // update
                int x = query[1], y = query[2], val = query[3];
                A.updateRange(x, y, val);
            } else {
                // answer
                int tot = query[1], res = 0;
                for (int x : nums1) {
                    int target = tot - x;
                    res += A.query(target);
                }
                ans.emplace_back(res);
            }
        }
        return ans;
    }
};
```

## 1340. Jump Game V

### Solution 1: dp, sorting, longest path in dag

You can jump from index i to index j only if:

arr[j] < arr[i]

and every element between them does not block the jump. You can jump left or right up to distance d, but you must stop scanning when you hit an element greater than or equal to arr[i].

The code sorts indices by decreasing value:

sort(idx.begin(), idx.end(), [&](int x, int y) {
    return arr[x] > arr[y];
});

Then it processes larger values first.

For each index i, it tries jumping to smaller reachable values on the right and left:

dp[i + j] = max(dp[i + j], dp[i] + 1);
dp[i - j] = max(dp[i - j], dp[i] + 1);

Here, dp[x] means the maximum number of visited indices in a valid jump sequence ending at index x.

Why sorting works:
All jumps go from a higher value to a lower value. That means the graph has no cycles. Sorting by decreasing height guarantees that when processing a node, its outgoing transitions go to lower-value nodes that should be updated after it.

Alternative view:
This is like longest path in a DAG, where each index is a node and valid jumps are directed edges from higher values to lower values.

```cpp
class Solution {
public:
    int maxJumps(vector<int>& arr, int d) {
        int N = arr.size();
        vector<int> dp(N, 1);
        vector<int> idx(N);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int x, int y) { return arr[x] > arr[y]; });
        for (int i : idx) {
            for (int j = 1; j <= d && i + j < N; ++j) {
                if (arr[i + j] >= arr[i]) break;
                dp[i + j] = max(dp[i + j], dp[i] + 1);
            }
            for (int j = 1; j <= d && i - j >= 0; ++j) {
                if (arr[i - j] >= arr[i]) break;
                dp[i - j] = max(dp[i - j], dp[i] + 1);
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};
```

# Leetcode Weekly Contest 504

## Q2. Maximum Number of Items From Sale I

### Solution 1: dp, 0/1 knapsack, divisor counting, sieve

Each item has:

a factor f
a price p

Buying an item gives value equal to the number of items whose factor is divisible by f.

So first, compute:

freq[f] = number of items with factor divisible by f

This is done with a multiples sieve:

for (int i = 1; i <= MAXN; ++i)
    for (int j = 2*i; j <= MAXN; j += i)
        freq[i] += freq[j];

After that, the problem becomes:

Pick some items under the budget to maximize total free-item value.

That is a classic 0/1 knapsack:

dp[b] = maximum sale value achievable using exactly or at most b budget

For each item:

dp[b] = max(dp[b], dp[b - p] + freq[f]);

Then after spending i budget in the DP, the remaining budget can always buy regular cheapest items at price base:

dp[i] + (budget - i) / base

So the final answer is:

max over all spent i:
    sale_items_from_dp + leftover_regular_items
Key idea

Use DP to decide which discount-triggering items are worth buying, then greedily spend leftover money on the cheapest normal item.

```cpp
const int MAXN = 1'505;
class Solution {
public:
    int maximumSaleItems(vector<vector<int>>& items, int budget) {
        int N = items.size();
        vector<int> freq(MAXN + 1, 0);
        int base = 1e9;
        for (const auto &item : items) {
            freq[item[0]]++;
            base = min(base, item[1]);
        }
        for (int i = 1; i <= MAXN; ++i) {
            for (int j = i + i; j <= MAXN; j += i) {
                freq[i] += freq[j];
            }
        }
        vector<int> dp(budget + 1, 0);
        for (const auto &item : items) {
            int f = item[0], p = item[1];
            for (int b = budget; b >= p; --b) {
                dp[b] = max(dp[b], dp[b - p] + freq[f]);
            }
        }
        int ans = budget / base;
        for (int i = 0; i <= budget; ++i) {
            ans = max(ans, dp[i] + (budget - i) / base);
        }
        return ans;
    }
};
```

## Q3. Maximum Number of Items From Sale II

### Solution 1: sorting, divisor counting, greedy, sieve

Again, compute:

freq[f] = number of items whose factor is divisible by f

So buying an item with factor f can unlock freq[f] related items.

Then sort items by price increasing.

Let:

base = cheapest item price

Normally, spending p money on regular cheapest items buys:

p / base

items approximately.

A sale item is only worth considering if it gives better value than buying cheapest items directly.

Since buying one sale item effectively gives itself plus some free related items, the code treats useful sale choices as worth roughly 2 items for price p.

So it only considers items where:

p < 2 * base

Because once:

p >= 2 * base

then buying this sale item is no better than using the same money to buy two cheapest items.

For each useful item:

cnt = freq[f] - 1

The -1 is because the purchased item itself should not be counted as a free duplicate.

Then:

take = min(budget / p, cnt)

Meaning:

cannot buy more than budget allows
cannot benefit from more copies than the number of eligible free items

Each taken item contributes:

2 * take

Then spend the remaining budget on cheapest items:

ans += budget / base;
Key idea

Buy only sale-triggering items that are strictly better than buying cheapest items normally. Once prices reach 2 * base, stop, because the sale is no longer profitable.

```cpp
class Solution {
public:
    int maximumSaleItems(vector<vector<int>>& items, int budget) {
        int N = items.size();
        vector<int> freq(N + 1, 0);
        for (const auto &item : items) {
            freq[item[0]]++;
        }
        for (int i = 1; i <= N; ++i) {
            for (int j = i + i; j <= N; j += i) {
                freq[i] += freq[j];
            }
        }
        // weakly increasing price
        sort(items.begin(), items.end(), [](const auto &x, const auto &y) {
            return x[1] < y[1];
        });
        int base = items[0][1], ans = 0;
        for (const auto &item : items) {
            int f = item[0], p = item[1];
            if (p >= 2 * base) break;
            int cnt = freq[f] - 1; // can'ta take iteself. 
            int threshold = budget / p;
            int take = min(threshold, cnt);
            ans += 2 * take;
            budget -= take * p;
        }
        ans += budget / base;
        return ans;
    }
};
```

## Q4. Lexicographically Maximum MEX Array

### Solution 1: mex, greedy, binary search, map, interval expansion

At each segment start, greedily create the shortest segment that achieves the largest possible MEX. This maximizes the current answer value while leaving as much array as possible for later segments.

```cpp
class Solution {
public:
    vector<int> maximumMEX(vector<int>& nums) {
        int N = nums.size();
        unordered_map<int, vector<int>> A;
        for (int i = 0; i < N; ++i) {
            A[nums[i]].emplace_back(i);
        }
        vector<int> ans;
        for (int l = 0; l < N;) {
            int r = l + 1;
            int mex = 0;
            while (true) {
                vector<int> &arr = A[mex];
                int j = lower_bound(arr.begin(), arr.end(), l) - arr.begin();
                if (j == arr.size()) {
                    break;
                }
                mex++;
                r = max(r, arr[j] + 1);
            }
            ans.emplace_back(mex);
            l = r;
        }
        return ans;
    }
};
```

# Leetcode Biweekly Contest 184

## Q2. Minimum Energy to Maintain Brightness

### Solution 1: line sweep, sorting

Compute union length of intervals and multiply by energy per covered index

```cpp
using int64 = long long;
class Solution {
public:
    int64 minEnergy(int n, int brightness, vector<vector<int>>& intervals) {
        int x = (brightness + 2) / 3;
        vector<pair<int, int>> events;
        int64 ans = 0;
        for (const auto &interv : intervals) {
            int l = interv[0], r = interv[1];
            events.emplace_back(l, 1);
            events.emplace_back(r + 1, -1);
        }
        sort(events.begin(), events.end());
        int cnt = 0, start = -1;
        for (const auto &[t, c] : events) {
            if (cnt == 0) {
                start = t;
            }
            cnt += c;
            if (cnt == 0) {
                ans += 1LL * x * (t - start);
            }
        }
        return ans;
    }
};
```

## Q3. Maximum Total Value of Covered Indices

### Solution 1: greedy, subarray minimum

Start with all 1s, then greedily improve each block using previous 0 and minimum value

```cpp
using int64 = long long;
const int INF = numeric_limits<int>::max();
class Solution {
public:
    int64 maxTotal(vector<int>& nums, string s) {
        int N = nums.size();
        int last = -1;
        int64 ans = 0;
        for (int i = 0; i < N; ++i) {
            if (s[i] == '1') ans += nums[i];
        }
        s += '0'; // padding
        int mn = INF;
        for (int i = 0; i < N; ++i) {
            if (s[i] == '0') {
                last = i;
                mn = nums[i];
            } else {
                mn = min(mn, nums[i]);
            }
            if (s[i] == '1' && s[i + 1] == '0') {
                if (last != -1) {
                    ans += max(0, nums[last] - mn);
                }
            }
        }
        return ans;
    }
};
```

## Q4. Maximum Score with Co-Prime Element
 
### Solution 1: smallest prime factor sieve, inclusion-exclusion, divisor sieve, frequency counting

For each candidate value, count numbers sharing prime factors using inclusion-exclusion

```cpp
using int64 = long long;
const int MAXN = 1e5 + 5, INF = numeric_limits<int>::max();
class Solution {
private:
    int spf[MAXN], freq[MAXN], div[MAXN];

    // nloglog(n)
    void sieve(int n) {
        for (int i = 0; i < n; i++) {
            spf[i] = i;
        }
        for (int64 i = 2; i < n; i++) {
            if (spf[i] != i) continue;
            for (int64 j = i * i; j < n; j += i) {
                if (spf[j] != j) continue;
                spf[j] = i;
            }
        }
    }

    vector<int> factorize(int x) {
        vector<int> factors;
        while (x > 1) {
            int p = spf[x];
            factors.emplace_back(p);
            while (x % p == 0) x /= p;
        }
        return factors;
    }
public:
    int maxScore(vector<int>& nums, int maxVal) {
        int ans = -INF;
        int N = nums.size();
        int maxV = maxVal;
        for (int x : nums) {
            maxV = max(maxV, x);
            freq[x]++;
        }
        for (int i = 1; i <= maxV; ++i) {
            for (int j = i; j <= maxV; j += i) {
                div[i] += freq[j];
            }
        }
        sieve(maxV + 1);
        for (int i = 1; i <= maxV; ++i) {
            if (i > maxVal && freq[i] == 0) continue;
            vector<int> primes = factorize(i);
            int sz = primes.size();
            int endMask = 1 << sz, cost = 0;
            for (int mask = 1; mask < endMask; ++mask) {
                int cnt = 0, cand = 1;
                for (int j = 0; j < sz; ++j) {
                    if ((mask >> j) & 1) {
                        cnt++;
                        cand *= primes[j];
                    }
                }
                if (cnt % 2 == 0) {
                    cost -= div[cand];
                } else {
                    cost += div[cand];
                }
            }
            if (freq[i] == 0) {
                cost = max(1, cost);
            } else if (i == 1) {
                cost = 0;
            } else {
                cost--;
            }
            ans = max(ans, i - cost);
        }
        return ans;
    }
};
```

# Leetcode Weekly Contest 505

## Q2. Valid Binary Strings With Cost Limit

### Solution 1: recursive backtracking, pruning, string generation

Generate strings recursively while stopping branches whose cost exceeds K

```cpp
class Solution {
private:
    int N, cost, K;
    vector<string> ans;
    string cur;
    void dfs() {
        if (cost > K) return;
        if (cur.size() == N) {
            if (cost <= K) ans.emplace_back(cur);
            return;
        }
        // 0
        cur += '0';
        dfs();
        cur.pop_back();
        // 1
        if (!cur.empty() && cur.back() == '1') return;
        cost += cur.size();
        cur += '1';
        dfs();
        cur.pop_back();
        cost -= cur.size();
    }
public:
    vector<string> generateValidStrings(int n, int k) {
        N = n, K = k, cost = 0;
        dfs();
        return ans;
    }
};
```

## Q3. Maximum Sum of M Non-Overlapping Subarrays I

### Solution 1: prefix sum, dp, monotonic deque

Use dp[k][i] and optimize valid subarray starts with a deque

```cpp
using int64 = long long;
const int64 INF = numeric_limits<int64>::max();
class Solution {
public:
    int64 maximumSum(vector<int>& nums, int m, int l, int r) {
        int N = nums.size();
        int64 ans = -INF;
        vector<int64> pref(N + 1, 0);
        for (int i = 0; i < N; ++i) {
            pref[i + 1] += pref[i] + nums[i];
        }
        vector<vector<int64>> dp(m + 1, vector<int64>(N + 1, -INF));
        for (int i = 0; i <= N; ++i) {
            dp[0][i] = 0;
        }
        for (int k = 1; k <= m; ++k) {
            deque<int> dq;
            for (int i = 1; i <= N; ++i) {
                dp[k][i] = dp[k][i - 1]; // take nothing
                int j = i - l;
                if (j >= 0 && dp[k - 1][j] != -INF) {
                    int64 cand = dp[k - 1][j] - pref[j];
                    while (!dq.empty() && cand >= dp[k - 1][dq.back()] - pref[dq.back()]) {
                        dq.pop_back();
                    }
                    dq.emplace_back(j);
                }
                // remove those that are no longer valid from front of dq
                while (!dq.empty() && dq.front() < i - r) {
                    dq.pop_front();
                }
                if (!dq.empty()) { // choose subarray ending at i - 1
                    int j = dq.front();
                    dp[k][i] = max(dp[k][i], pref[i] + dp[k - 1][j] - pref[j]);
                }
            }
            ans = max(ans, dp[k][N]);
        }
        return ans;
    }
};
```

## Q4. Maximum Sum of M Non-Overlapping Subarrays II

### Solution 1: Lagrangian Relaxation, Alien's Trick, binary search, monotonic deque, 

Penalize each chosen subarray, binary search penalty, recover exact-m answer

Important subtlety

Even if the returned cnt is less than m, this technique can still recover the exact constrained answer under the usual convexity/monotonicity conditions of WQS binary search.

The binary search finds a supporting penalty value where the penalized objective encodes the best exact-m value. The + lambda * m operation evaluates the corresponding line at m.

This is the part that makes it feel unintuitive: the returned solution of the penalized problem does not always literally choose exactly m, but the transformed value can still expose the exact-m optimum.

Key technique

This is:

DP + monotonic deque + Lagrangian relaxation + binary search on penalty

Also known as:

Aliens trick
WQS binary search

```cpp
using int64 = long long;
const int64 INF = numeric_limits<int64>::max();
class Solution {
private:
    vector<int64> pref;
    pair<int64, int> solve_penalty(int l, int r, int64 lambda) {
        int N = pref.size() - 1;
        vector<int64> dp(N + 1, -INF);
        vector<int> cnt(N + 1, 0);
        deque<tuple<int, int64, int>> dq;
        for (int i = 1; i <= N; ++i) {
            dp[i] = dp[i - 1];
            cnt[i] = cnt[i - 1];
            int j = i - l;
            if (j >= 0) {
                int64 cand = dp[j];
                int count = cnt[j];
                if (cand < 0) {
                    cand = 0;
                    count = 0;
                }
                cand -= pref[j];
                while (!dq.empty()) {
                    auto [x, y, z] = dq.back();
                    if (make_pair(cand, -count) < make_pair(y, -z)) break;
                    dq.pop_back();
                }
                dq.emplace_back(j, cand, count);
            }
            // remove those that are no longer valid from front of dq
            while (!dq.empty() && get<0>(dq.front()) < i - r) {
                dq.pop_front();
            }
            if (!dq.empty()) { // choose subarray ending at i - 1
                auto [j, ncost, ncount] = dq.front();
                ncost += pref[i] - lambda;
                ncount++;
                if (ncost > dp[i] || (ncost == dp[i] && ncount < cnt[i])) {
                    dp[i] = ncost;
                    cnt[i] = ncount;
                }
            }
        }
        return {dp[N], cnt[N]};
    }
public:
    int64 maximumSum(vector<int>& nums, int m, int l, int r) {
        int N = nums.size();
        pref.assign(N + 1, 0);
        for (int i = 0; i < N; ++i) {
            pref[i + 1] += pref[i] + nums[i];
        }
        int64 lo = 0, hi = 1e18;
        while (lo < hi) {
            int64 mid = lo + (hi - lo) / 2;
            auto [penalizedCost, cnt] = solve_penalty(l, r, mid);
            if (cnt > m) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        auto [penalizedCost, cnt] = solve_penalty(l, r, lo);
        int64 ans = penalizedCost + 1LL * lo * m;
        return ans;
    }
};
```

## 3691. Maximum Total Subarray Value II

### Solution 1: maxheap, two pointer

```cpp
using int64 = long long;

template <class T, class Op>
struct SparseTable {
    vector<int> lg;          // floor(log2(i))
    vector<vector<T>> st;    // st[k][i] = combine over [i, i + 2^k)

    SparseTable() = default;
    explicit SparseTable(const vector<T>& a) { build(a); }

    void build(const vector<T>& a) {
        int n = a.size();
        lg.assign(n + 1, 0);
        for (int i = 2; i <= n; ++i) lg[i] = lg[i / 2] + 1;

        int K = (n == 0) ? 0 : (lg[n] + 1);
        st.assign(K, vector<T>(n));
        if (n == 0) return;

        st[0] = a;
        for (int k = 1; k < K; ++k) {
            int len = 1 << k;
            int half = len >> 1;
            for (int i = 0; i + len <= n; ++i) {
                st[k][i] = Op::combine(st[k - 1][i], st[k - 1][i + half]);
            }
        }
    }

    // Query on half-closed interval [l, r]
    // Precondition: 0 <= l <= r <= n
    T query(int l, int r) const {
        int len = r - l + 1;
        int k = lg[len];
        // Two blocks cover [l,r], potentially overlapping: OK only if idempotent.
        return Op::combine(st[k][l], st[k][r - (1 << k) + 1]);
    }
};

// ----- Plug-in ops (examples) -----

template <class T>
struct MinOp {
    static T identity() {
        return numeric_limits<T>::max();
    }

    static T combine(T a, T b) {
        return min(a, b);
    }
};
template <class T>
struct MaxOp {
    static T identity() {
        return numeric_limits<T>::lowest();
    }

    static T combine(T a, T b) {
        return max(a, b);
    }
};
class Solution {
public:
    int64 maxTotalValue(vector<int>& nums, int k) {
        int N = nums.size();
        SparseTable<int, MinOp<int>> stmin(nums);
        SparseTable<int, MaxOp<int>> stmax(nums);
        int64 ans = 0;
        priority_queue<tuple<int, int, int>> maxheap;
        for (int i = 0; i < N; ++i) {
            int val = stmax.query(i, N - 1) - stmin.query(i, N - 1);
            maxheap.emplace(val, i, N - 1);
        }
        while (k--) {
            auto [val, l, r] = maxheap.top();
            maxheap.pop();
            ans += val;
            if (l == r) continue;
            int nextVal = stmax.query(l, r - 1) - stmin.query(l, r - 1);
            maxheap.emplace(nextVal, l, r - 1);
        }
        return ans;
    }
};
```