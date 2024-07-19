# Atcoder Beginner Contest 362

## E - Count Arithmetic Subsequences 

### Solution 1:  dynamic programming, counting, arithmetic sequences, coordinate compression

```cpp
const int MOD = 998244353;
int N;
vector<int> arr;
vector<vector<vector<int>>> dp; // (i, k, d)

void solve() {
    cin >> N;
    arr.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    vector<int> diff;
    for (int i = 1; i < N; i++) {
        for (int j = 0; j < i; j++) {
            diff.push_back(arr[i] - arr[j]);
        }
    }
    // coordinate compression
    sort(diff.begin(), diff.end());
    diff.erase(unique(diff.begin(), diff.end()), diff.end());
    map<int, int> index;
    for (int i = 0; i < diff.size(); i++) {
        index[diff[i]] = i;
    }
    dp.assign(N, vector<vector<int>>(N, vector<int>(diff.size(), 0)));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < diff.size(); j++) {
            dp[i][0][j] = 1;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            for (int k = 1; k < N; k++) {
                int d = arr[i] - arr[j]; // j < i
                int idx = index[d];
                dp[i][k][idx] = (dp[i][k][idx] + dp[j][k - 1][idx]) % MOD;
            }
        }
    }
    cout << N << " ";
    for (int k = 1; k < N; k++) {
        int ans = 0;
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < diff.size(); d++) {
                ans = (ans + dp[i][k][d]) % MOD;
            }
        }
        cout << ans << " ";
    }
    cout << endl;
}

signed main() {
    solve();
    return 0;
}
```

## F - Perfect Matching on a Tree 

### Solution 1: 

```cpp

```

## G - Count Substring Query 

### Solution 1:  suffix array with radix sort, binary search, memoize solution speedup, offline queries

```cpp
int Q;
string S, T;
vector<int> bucket_size, bucket_pos, leaderboard, update_leaderboard, equivalence_class, update_equivalence_class;

void radix_sort() {
    int n = leaderboard.size();
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

vector<int> suffix_array(string& s) {
    int n = s.size();
    vector<pair<char, int>> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = {s[i], i};
    }
    sort(arr.begin(), arr.end());
    leaderboard.assign(n, 0);
    equivalence_class.assign(n, 0);
    for (int i = 0; i < n; i++) {
        leaderboard[i] = arr[i].second;
    }
    equivalence_class[leaderboard[0]] = 0;
    for (int i = 1; i < n; i++) {
        int left_segment = arr[i - 1].first;
        int right_segment = arr[i].first;
        equivalence_class[leaderboard[i]] = equivalence_class[leaderboard[i - 1]] + (left_segment != right_segment);
    }
    bool is_finished = false;
    int k = 1;
    while (k < n && !is_finished) {
        for (int i = 0; i < n; i++) {
            leaderboard[i] = (leaderboard[i] - k + n) % n; // create left segment, keeps sort of the right segment
        }
        radix_sort(); // radix sort for the left segment
        swap(leaderboard, update_leaderboard);
        update_equivalence_class.assign(n, 0);
        update_equivalence_class[leaderboard[0]] = 0;
        for (int i = 1; i < n; i++) {
            pair<int, int> left_segment = {equivalence_class[leaderboard[i - 1]], equivalence_class[(leaderboard[i - 1] + k) % n]};
            pair<int, int> right_segment = {equivalence_class[leaderboard[i]], equivalence_class[(leaderboard[i] + k) % n]};
            update_equivalence_class[leaderboard[i]] = update_equivalence_class[leaderboard[i - 1]] + (left_segment != right_segment);
            is_finished &= (update_equivalence_class[leaderboard[i]] != update_equivalence_class[leaderboard[i - 1]]);
        }
        k <<= 1;
        swap(equivalence_class, update_equivalence_class);
    }
    return leaderboard;
}

int binary_search(string target) {
    int lo = 0, hi = S.size();
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (S.substr(leaderboard[mid], target.size()) < target) lo = mid + 1;
        else hi = mid;
    }   
    return lo;
}

map<string, vector<int>> queries;
vector<int> ans;


void solve() {
    cin >> S;
    cin >> Q;
    S += "$";
    suffix_array(S);
    ans.resize(Q);
    for (int i = 0; i < Q; i++) {
        cin >> T;
        if (queries.find(T) == queries.end()) queries[T] = vector<int>();
        queries[T].push_back(i);
    }
    for (auto [T, indices] : queries) {
        int l = binary_search(T), r = binary_search(T + '~');
        for (int i : indices) {
            ans[i] = r - l;
        }
    }
    for (int x : ans) {
        cout << x << endl;
    }
}

signed main() {
    solve();
    return 0;
}
```