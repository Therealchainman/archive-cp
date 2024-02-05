# Leetcode Weekly Contest 383

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## 3031. Minimum Time to Revert Word to Initial State II

### Solution 1:  dynamic programming, z algorithm, string matching, substring matching prefix

```py
def z_algorithm(s: str) -> list[int]:
    n = len(s)
    z = [0]*n
    left = right = 0
    for i in range(1,n):
        # BEYOND CURRENT MATCHED SEGMENT, TRY TO MATCH WITH PREFIX
        if i > right:
            left = right = i
            while right < n and s[right-left] == s[right]:
                right += 1
            z[i] = right - left
            right -= 1
        else:
            k = i - left
            # IF PREVIOUS MATCHED SEGMENT IS NOT TOUCHING BOUNDARIES OF CURRENT MATCHED SEGMENT
            if z[k] < right - i + 1:
                z[i] = z[k]
            # IF PREVIOUS MATCHED SEGMENT TOUCHES OR PASSES THE RIGHT BOUNDARY OF CURRENT MATCHED SEGMENT
            else:
                left = i
                while right < n and s[right-left] == s[right]:
                    right += 1
                z[i] = right - left
                right -= 1
    return z

class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        z_arr = z_algorithm(word)
        for i in range(k, n, k):
            len_ = n - i
            if z_arr[i] == len_: return i // k
        return math.ceil(n / k)
```

### Solution 2:  suffix arrays, longest common prefix (lcp), radix sort

```cpp
std::vector<int> radix_sort(const std::vector<int>& leaderboard, const std::vector<int>& equivalence_class) {
    int n = leaderboard.size();
    std::vector<int> bucket_size(n, 0);
    for (int eq_class : equivalence_class) {
        bucket_size[eq_class]++;
    }

    std::vector<int> bucket_pos(n, 0);
    for (int i = 1; i < n; ++i) {
        bucket_pos[i] = bucket_pos[i-1] + bucket_size[i-1];
    }

    std::vector<int> updated_leaderboard(n, 0);
    for (int i = 0; i < n; ++i) {
        int eq_class = equivalence_class[leaderboard[i]];
        int pos = bucket_pos[eq_class];
        updated_leaderboard[pos] = leaderboard[i];
        bucket_pos[eq_class]++;
    }

    return updated_leaderboard;
}

std::pair<std::vector<int>, std::vector<int>> suffix_array(const std::string& s) {
    int n = s.size();
    std::vector<std::pair<char, int>> arr(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = std::make_pair(s[i], i);
    }

    std::sort(arr.begin(), arr.end());

    std::vector<int> leaderboard(n, 0), equivalence_class(n, 0);
    for (int i = 0; i < n; ++i) {
        leaderboard[i] = arr[i].second;
    }

    equivalence_class[leaderboard[0]] = 0;
    for (int i = 1; i < n; ++i) {
        equivalence_class[leaderboard[i]] = equivalence_class[leaderboard[i-1]] + (arr[i].first != arr[i-1].first);
    }

    bool is_finished = false;
    int k = 1;
    while (k < n && !is_finished) {
        is_finished = true; // Corrected initialization within loop
        for (int i = 0; i < n; ++i) {
            leaderboard[i] = (leaderboard[i] - k + n) % n;
        }

        leaderboard = radix_sort(leaderboard, equivalence_class);

        std::vector<int> updated_equivalence_class(n, 0);
        updated_equivalence_class[leaderboard[0]] = 0;
        for (int i = 1; i < n; ++i) {
            std::pair<int, int> left_segment = std::make_pair(equivalence_class[leaderboard[i-1]], equivalence_class[(leaderboard[i-1]+k)%n]);
            std::pair<int, int> right_segment = std::make_pair(equivalence_class[leaderboard[i]], equivalence_class[(leaderboard[i]+k)%n]);
            updated_equivalence_class[leaderboard[i]] = updated_equivalence_class[leaderboard[i-1]] + (left_segment != right_segment);
            is_finished &= (updated_equivalence_class[leaderboard[i]] == updated_equivalence_class[leaderboard[i-1]]);
        }

        k <<= 1;
        equivalence_class = std::move(updated_equivalence_class);
    }

    return {leaderboard, equivalence_class};
}

std::vector<int> lcp(const std::vector<int>& leaderboard, const std::vector<int>& equivalence_class, const std::string& s) {
    int n = s.size();
    std::vector<int> lcp(n-1, 0);
    int k = 0;
    for (int i = 0; i < n-1; ++i) {
        int pos_i = equivalence_class[i];
        int j = leaderboard[pos_i - 1];
        while (s[(i + k) % n] == s[(j + k) % n]) {
            k++;
        }
        lcp[pos_i-1] = k;
        k = std::max(k - 1, 0);
    }

    return lcp;
}

class Solution {
public:
    int minimumTimeToInitialState(string word, int k) {
        int n = word.size();
        std::string augmentedWord = word + "$"; // Append '$' to the word
        auto [leaderboard, eq_class] = suffix_array(augmentedWord);
        std::vector<int> lcp_arr = lcp(leaderboard, eq_class, augmentedWord);

        int pivot = 0;
        for (int i = 0; i <= n; ++i) {
            if (leaderboard[i] == 0) {
                pivot = i;
                break;
            }
        }

        int ans = std::ceil(static_cast<double>(n) / k);
        int min_lcp = n;
        // Check to the left of the pivot
        for (int i = pivot - 1; i > 0; --i) {
            if (lcp_arr[i] < min_lcp) {
                min_lcp = lcp_arr[i];
            }
            if (leaderboard[i] % k != 0) continue;
            int len_ = n - leaderboard[i];
            if (len_ > min_lcp) continue;
            ans = std::min(ans, leaderboard[i] / k);
            if (ans == 1) return ans;
        }

        min_lcp = n;
        // Check to the right of the pivot
        for (int i = pivot + 1; i <= n; ++i) {
            if (lcp_arr[i - 1] < min_lcp) {
                min_lcp = lcp_arr[i - 1];
            }
            if (leaderboard[i] % k != 0) continue;
            int len_ = n - leaderboard[i];
            if (len_ > min_lcp) continue;
            ans = std::min(ans, leaderboard[i] / k);
            if (ans == 1) return ans;
        }

        return ans;
    }
};
```

### Solution 3:  KMP algorithm, longest common prefix (lcp)

```py

```
