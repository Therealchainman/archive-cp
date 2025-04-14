# Rolling Median Deviation

## The algorithm in python

This is the algorithm that uses prefix sums and is O(n) time complexity.  For this specific example it is calculating the smallest median deviation over the array, can also find the maximum if you just swap that out.  Or even find the sum, don't know why you'd want that though.

Note this is for a median when you take subarray of size k

```py
n = len(nums)
nums.sort()
psum = list(accumulate(nums))
def sum_(i, j):
    return psum[j] - (psum[i - 1] if i > 0 else 0)
def deviation(i, j, mid):
    lsum = (mid - i + 1) * nums[mid] - sum_(i, mid)
    rsum = sum_(mid, j) - (j - mid + 1) * nums[mid]
    return lsum + rsum
def RMD(nums, k): # rolling median deviation
    ans = math.inf
    l = 0
    for r in range(k - 1, n):
        mid = (l + r) >> 1
        ans = min(ans, deviation(l, r, mid))
        if k % 2 == 0:
            ans = min(ans, deviation(l, r, mid + 1))
        l += 1
    return ans
```

This is useful, because when you want to calculate the best value to set all elements to in a fixed sized subarray, the median is the best option, can be proved because if you move away from the median, it will increase more elements than those it decreases. 


## Fixed sized sliding window

The RMD struct efficiently calculates the sum of absolute differences from the median for every prefix of a given array arr, and for a sliding window of size k if desired.

Calculates the sum of the absolute difference from the median for the ith index in result, and contains k elements, so goes back i - k + 1

This one is 1-indexed also.  So result[3] = elements from 0,1,2 index are in there. 

This algorithm implements a sliding window median cost computation, often used in range median queries or optimization problems. Here's a concise summary of its core functionality:

Given an array arr and a window size k, for each position i (0-based), compute the total absolute deviation from the median of the k elements in the sliding window ending at i.

```cpp
using int64 = long long;
const int64 INF = (1LL << 63) - 1;
struct RMD {
    vector<int64> result;
    multiset<int64> left, right;
    int64 leftSum, rightSum;
    void init(const vector<int>& arr, int k) {
        int N = arr.size();
        leftSum = rightSum = 0;
        result.assign(N + 1, 0);
        for (int i = 0; i < N; i++) {
            add(arr[i]);
            int64 median = *prev(left.end());
            int64 cost = median * left.size() - leftSum + rightSum - median * right.size();
            result[i + 1] = cost;
            if (i >= k - 1) {
                remove(arr[i - k + 1]);
            }
        }
    }
    void balance() {
        while (left.size() > right.size() + 1) {
            auto it = prev(left.end());
            int val = *it;
            leftSum -= val;
            left.erase(it);
            rightSum += val;
            right.insert(val);
        }
        while (left.size() < right.size()) {
            auto it = right.begin();
            int val = *it;
            rightSum -= val;
            right.erase(it);
            leftSum += val;
            left.insert(val);
        }
    }
    void add(int num) {
        if (left.empty() || num <= *prev(left.end())) {
            left.insert(num);
            leftSum += num;
        } else {
            right.insert(num);
            rightSum += num;
        }
        balance();
    }
    void remove(int num) {
        if (left.find(num) != left.end()) {
            auto it = left.find(num);
            int64 val = *it;
            leftSum -= val;
            left.erase(it);
        } else {
            auto it = right.find(num);
            int64 val = *it;
            rightSum -= val;
            right.erase(it);
        }
        balance();
    }
};
```

## Rolling Medians for calculating difference between larger half and smaller half

MedianBalancer is a data structure that dynamically maintains the balanced partition of a prefix of integers into two halves (lower and upper), and for every even-length prefix, it computes the difference between the sum of the upper half and the lower half.

result[i]=sum of larger half−sum of smaller half

```cpp
struct MedianBalancer {
    vector<int64> result;
    multiset<int64> left, right;
    int64 leftSum, rightSum;
    void init(const vector<int>& arr, int k) {
        int N = arr.size();
        leftSum = rightSum = 0;
        result.assign(N + 1, 0);
        for (int i = 0; i < N; i++) {
            if (i % 2 == 0) {
                result[i] = rightSum - leftSum;;
            }
            add(arr[i]);
        }
    }
    void balance() {
        while (left.size() > right.size() + 1) {
            auto it = prev(left.end());
            int val = *it;
            leftSum -= val;
            left.erase(it);
            rightSum += val;
            right.insert(val);
        }
        while (left.size() < right.size()) {
            auto it = right.begin();
            int val = *it;
            rightSum -= val;
            right.erase(it);
            leftSum += val;
            left.insert(val);
        }
    }
    void add(int num) {
        if (left.empty() || num <= *prev(left.end())) {
            left.insert(num);
            leftSum += num;
        } else {
            right.insert(num);
            rightSum += num;
        }
        balance();
    }
    void remove(int num) {
        if (left.find(num) != left.end()) {
            auto it = left.find(num);
            int64 val = *it;
            leftSum -= val;
            left.erase(it);
        } else {
            auto it = right.find(num);
            int64 val = *it;
            rightSum -= val;
            right.erase(it);
        }
        balance();
    }
};
```