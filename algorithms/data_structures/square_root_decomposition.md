# Square Root Decomposition

General guidelines for when to use square root decomposition:

1. An array or sequence.
2. Many queries/updates.
3. A range [l, r] is involved.
4. Brute force per query is too slow.
5. You can summarize each block in a way that helps.

## Example implementation of Square Root Decompositionhis

This can be repurposed

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

Square root decomposition is the simplest “preprocess to answer fast” pattern

With sqrt decomposition for RMQ, you split the array into blocks of size ~√n, store each block’s minimum, then answer a query by:
- scanning the partial block on the left,
- scanning the partial block on the right,
- taking mins of full blocks in the middle.

That immediately makes the core concept concrete:
- precompute summaries
- pay a bit up front
- queries get faster
- you can tune block size to balance build vs query time

Sparse Table is also preprocessing, but it’s less intuitive at first because the preprocessing is “all powers of two” and the query trick is “two overlapping intervals”.

## Square Root with small k/large k trick

You use two different algorithms for the queries with small and large k. 

For example, you use a naive approach for large k, while for small k, you use a more sophisticated method involving residue-class difference arrays. (just for example)

