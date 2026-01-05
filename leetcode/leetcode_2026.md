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

##

### Solution 1: 

```cpp

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

##

### Solution 1: 

```ts

```

##

### Solution 1: 

```ts

```

##

### Solution 1: 

```ts

```