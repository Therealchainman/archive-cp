# Eolymp

# Eolymp Weekend Practice #5 

## Sacrifice

### Solution 1: sorting, greedy, pointers, modular arithmetic

```cpp
int main(){
    int t;cin>>t;
    while(t--){
        int n,m,k;cin>>n>>m>>k;
        vector<int>a(n);
        for(int i=0;i<n;i++){
            cin>>a[i];
        }
        int p = 0;
        m--;
        int suffixLen = n - m;
        int remK = k - suffixLen;
        if (remK > 0) {
            p = remK % n;
        }
        sort(a.begin(), a.end());
        vector<int> ans(n, 0);
        for (int i = m; i < p; i++) {
            ans[i] = a.back();
            a.pop_back();
        }
        for (int i = max(m, p); i < n; i++) {
            ans[i] = a.back();
            a.pop_back();
        }
        int i = 0;
        while (!a.empty()) {
            ans[i++] = a.back();
            a.pop_back();
        }
        for (int x : ans) {
            cout << x << " ";
        }
        cout << endl;
    }
    return 0;
}
```

## Score

### Solution 1:  fenwick tree for counting prefix sums, math flor and ceil, binary search, prefix sum, sorting, map, coordinate compression

```cpp
const int64 INF = 1e18;
template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : T(0);
    }
};
int64 ceil(int64 x, int64 y) {
    return (x + y - 1) / y;
}
int64 floor(int64 x, int64 y) {
    return (x / y) - ((x % y) != 0 && (x < 0) != (y < 0));
}
int main(){
    int t;cin>>t;
    while(t--){
        int n;cin>>n;
        vector<int64>a(n);
        for(int i=0;i<n;i++){
            cin>>a[i];
        }
        int64 totalSum = accumulate(a.begin(), a.end(), 0LL);
        int64 psum = 0;
        vector<int64> indices = {0};
        for (int64 x : a) {
            psum += x;
            indices.emplace_back(psum);
        }
        sort(indices.begin(), indices.end());
        indices.erase(unique(indices.begin(), indices.end()), indices.end());
        indices.insert(indices.begin(), -INF);
        map<int64, int> valueToIndex;
        int M = indices.size();
        for (int i = 0; i < M; i++) {
            valueToIndex[indices[i]] = i;
        }
        FenwickTree<int64> ft;
        ft.init(M);
        ft.update(valueToIndex[0], 1);
        map<int64, int> freq;
        freq[0] = 1;
        psum = 0;
        int64 ans = 0;
        for (int i = 0; i < n; i++) {
            psum += a[i];
            int64 c1 = floor(2LL * psum - totalSum, 2);
            int r = upper_bound(indices.begin(), indices.end(), c1) - indices.begin() - 1;
            int64 cntWonderful = ft.query(r);
            int64 target = psum - floor(totalSum, 2);
            ans += 2 * cntWonderful;
            ans -= i + 1 - cntWonderful;
            if (totalSum % 2 == 0) ans += freq[target];
            freq[psum]++;
            ft.update(valueToIndex[psum], 1);
        }
        cout << ans << endl;
    }
    return 0;
}
```