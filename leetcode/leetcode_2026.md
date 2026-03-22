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

### Solution 1:

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

##

### Solution 1:

```cpp

```




