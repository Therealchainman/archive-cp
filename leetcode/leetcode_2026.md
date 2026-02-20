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
            if (bal == 0) {
                string inner = s.substr(j + 1, i - j - 1);
                specialArr.emplace_back('1' + makeLargestSpecial(inner) + '0');
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
