# Februrary Contest 2022

## Robot Instructions

### Solution 1:  meet in the middle, binary search for lower and upper bounds

O(N * 2^N/2)

```cpp
const int MAXN = 41;
int N, cnt;
int xg, yg, x, y;
int ans[MAXN];
vector<pair<int, int>> arr1, arr2;
vector<pair<int, int>> pool[MAXN]; 
pair<int, int> target;

void solve() {
    cin >> N;
    cin >> xg >> yg;
    int n1 = N / 2 + 1;
    int n2 = N - n1;
    for (int i = 0; i < N; i++) {
        cin >> x >> y;
        if (i < n1) {
            arr1.emplace_back(x, y);
        } else {
            arr2.emplace_back(x, y);
        }
    }
    for (int mask = 0; mask < (1 << n1); mask++) {
        x = 0, y = 0, cnt = 0;
        for (int i = 0; i < n1; i++) {
            if ((mask >> i) & 1) {
                x += arr1[i].first;
                y += arr1[i].second;
                cnt++;
            }
        }
        pool[cnt].emplace_back(x, y);
    }
    for (int i = 1; i <= n1; i++) {
        sort(pool[i].begin(), pool[i].end());
    }
    memset(ans, 0, sizeof(ans));
    for (int mask = 0; mask < (1 << n2); mask++) {
        x = 0, y = 0, cnt = 0;
        for (int i = 0; i < n2; i++) {
            if ((mask >> i) & 1) {
                x += arr2[i].first;
                y += arr2[i].second;
                cnt++;
            }
        }
        target = make_pair(xg - x, yg - y);
        for (int i = 0; i <= n1; i++) {
            int lb = lower_bound(pool[i].begin(), pool[i].end(), target) - pool[i].begin();
            int rb = upper_bound(pool[i].begin(), pool[i].end(), target) - pool[i].begin();
            ans[cnt + i] += rb - lb;
        }
    }
    for (int i = 1; i <= N; i++) {
        cout << ans[i] << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    while (T--) {
        solve();
    }
    return 0;
}

```