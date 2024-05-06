# Subset Sum Problem

Example code for how to solve the minimum number of elements needed to sum to a each target value.  This is code that can be ran in O(sqrt(N) * N) depending on the size of one distinct elements.  This is cool I guess.  Another thing is this assumes you can only take one of each distinct element. 

This is actually solving for the subset sum problem when you consider a subset of just the elements arr[0...i], that is you are just working with the first i elements in the array, and you can only take from those to create the sum from a subset of those i elements.  

This update represents the scenario where the current cycle is included in the subset (the "+1" accounts for including the current cycle). Similarly, it updates dp1[i][j] to ensure it carries forward the minimum number of elements needed to achieve sum j without including the current cycle. This step encapsulates the choice at the heart of the subset sum problem: to include or not include the current element in a particular sum.

```cpp
vector<vector<int>> dp1(cycles.size(), vector<int>(N + 1, INF));
dp1[0][0] = 0;
for (int i = 0; i < cycles.size(); i++) {
    int c = cycles[i];
    if (i == 0) {
        dp1[i][c] = 1;
        continue;
    }
    for (int j = 0; j <= N - c; j++) {
        dp1[i][j + c] = min(dp1[i][j + c], dp1[i - 1][j] + 1); // take the cycle
        dp1[i][j] = min(dp1[i][j], dp1[i - 1][j]); // don't take the cycle
    }
}
```

Imagine a subset sum problem where you are dealing with something more like a multiset.  You have multiples of each value. so now it could be O(N^2), but there is a speed up to get it to be O(sqrt(N)*N), and let's say you still want to compute the minimum number of elements need to sum to each target value. 

This algorithm is for when you have dp2[x] = minimum size of a subset that sums to x.  Just note that you are not doing this for just first i elements, it's for the entire set. 

For this specific example you are taking freq[i] - 1, but you could just as well not do that.  And you know that the sum of freq[i] <= N, this can also be solved in O(sqrt(N)*N)

```cpp
vector<int> dp2(N + 1, INF);
dp2[0] = 0;
// Now we need to do something similar but for if you take freq - 1 of each distinct element.
for (int c : cycles) {
    int f = freq[c] - 1;
    if (f == 0) continue;
    for (int j = 0; j < c; j++) {
        int mn = INF;
        vector<array<int, 2>> v;
        int i = 0, r = 0;
        for (int k = j; k <= N; k += c, r++) {
            while (i < v.size()) {
                auto [x, y] = v[i];
                if (y >= r - f) break;
                i++;
            }
            int mn = dp2[k];
            if (i < v.size()) {
                dp2[k] = min(dp2[k], v[i][0] + r - v[i][1]);
            }
            while (v.size()) {
                auto [x, y] = v.back();
                if (x - y >= mn - r) v.pop_back();
                else break;
            }
            v.push_back({mn, r});
            i = min(i, (int)v.size() - 1);
        }
    }
}
```