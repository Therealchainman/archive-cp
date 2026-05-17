# suffix array

Suffix array = sorted list of suffix starting positions
LCP array = for each neighboring pair in that sorted list, the length of their common prefix

### C++ implementation of suffix array

Suffix array is an array of integers, where the integers represent the suffix from a string.
the integer in suffix array represents the starting index for the suffix. 
suffix array is these suffix index sorted in order of suffix order from ascending order

sorting is O(n+k) where k is the range of values in the string.

To use this you must append the "$" character at end of the string, to get the suffix array sorted.

```cpp
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

/*
how to call suffix array, then use the leaderboard
string S + "$";
suffix_array(S);
and if you using lcp, make sure to remove the last
character in the suffix array and the lcp array, because they are for the "$" character, which is not part of the original string.
*/
```

## suffix array notes

![image](images/suffix_array_and_lcp/suffix_array_1.png)
![image](images/suffix_array_and_lcp/suffix_array_2.png)
![image](images/suffix_array_and_lcp/suffix_array_3.png)
![image](images/suffix_array_and_lcp/lcp_array_1.png)
![image](images/suffix_array_and_lcp/lcp_array_2.png)

## Longest repeated substring

This one can be solved with suffix array and LCP array. The time complexity is O(nlogn) and space complexity is O(n). 

Uses the code above for suffix and lcp array

Find the adjacent suffix pair with the largest common prefix. That common prefix is the longest duplicated substring.

```py
def longestDupSubstring(self, s: str) -> str:
    s += '$'
    n = len(s)
    p, c = suffix_array(s)
    lcp_arr = lcp(p, c, s)
    idx = max(range(n - 1), key = lambda i: lcp_arr[i])
    len_ = lcp_arr[idx]
    suffix_index = p[idx]
    return s[suffix_index: suffix_index + len_]
```

## notes for longest repeated and non overlapping substring

At first I tried to get suffix array and lcp array to work to solve this problem.  But I found contradictions that lead me to believe it doens't work.  And the best solution is dynamic programming that is relatively easy to learn.

![image](images/repeating_nonoverlapping_substrings/repeating_nonoverlapping_substrings_1.png)
![image](images/repeating_nonoverlapping_substrings/repeating_nonoverlapping_substrings_2.png)
![image](images/repeating_nonoverlapping_substrings/repeating_nonoverlapping_substrings_3.png)
![image](images/repeating_nonoverlapping_substrings/repeating_nonoverlapping_substrings_4.png)
![image](images/repeating_nonoverlapping_substrings/repeating_nonoverlapping_substrings_5.png)

## Longest repeated and non-overlapping substring

Dynamic programming with time complexity of O(n^2) can solve this one. 

dp[i][j] is the longest common substring with both substrings ending at ith and jth character. The transition is from the i-1 j-1 end character.  So if the current characters are equal and the length doesn't cause overlap of the substrings then it's good. 


```py
def longestSubstring(self, S , N):
    dp = [[0]*(N + 1) for _ in range(N + 1)]
    max_len = 0
    res = ''
    for i in range(N):
        for j in range(i + 1, N):
            if S[i] == S[j] and dp[i][j] < j - i:
                dp[i + 1][j + 1] = max(dp[i + 1][j + 1], dp[i][j] + 1)
                if dp[i + 1][j + 1] > max_len:
                    max_len = dp[i + 1][j + 1]
                    res = S[i - max_len + 1 : i + 1]
    return res if len(res) > 0 else -1
```

## Given multiple patterns matching to a string text

This can be used to solve problems involving you have Q queries, where each query is a string.
Then you want to find the count or the positions in some text string S, where those queries are a substring of the text.
