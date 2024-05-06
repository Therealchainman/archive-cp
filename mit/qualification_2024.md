# MIT Qualification 2024

## 3-SAT 

### Solution 1: 

```py
from collections import Counter
def main():
    N, M = map(int, input().split())
    outdegrees = Counter()
    for i in range(M):
        x, y, z = map(int, input().split())
        x -= 1; y -= 1; z -= 1
        clause = tuple(sorted(set([x, y, z])))
        outdegrees[clause] += 1
    ans = [0] * N
    if M & 1:
        print("YES")
        print(*[1] * N)
    else:
        for c, v in outdegrees.items():
            if v & 1:
                print("YES")
                for i in c:
                    ans[i] = 1
                print(*ans)
                return
        print("NO")
    
if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## Busy Marksman

### Solution 1:  greedy, pick from lanes with one target immediately, else pick from rest

```cpp
const int MAXN = 300'005, MAXM = 500'005;
int N, A[MAXN], ans[MAXM];

void solve() {
    cin >> N;
    int sum = 0;
    vector<int> rest, ones;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        sum += A[i];
        if (A[i] == 1) ones.push_back(i);
        else if (A[i] > 0) rest.push_back(i);
    }
    int i;
    for (i = 1; i <= sum; i++) {
        if (i & 1) {
            if (ones.size()) {
                int v = ones.end()[-1];
                ones.pop_back();
                ans[i - 1] = v + 1;
                A[v]--;
            } else if (rest.size()) {
                int v = rest.end()[-1];
                ans[i - 1] = v + 1;
                A[v]--;
                if (A[v] == 1) {
                    ones.push_back(v);
                    rest.pop_back();
                }
            } else {
                break;
            }
        } else {
            if (rest.size()) {
                int v = rest.end()[-1];
                ans[i - 1] = v + 1;
                A[v]--;
                if (A[v] == 1) {
                    ones.push_back(v);
                    rest.pop_back();
                }
            } else {
                break;
            }
        }
    }

    if (i > sum) {
        cout << "YES" << endl;
        for (int j = 0; j < sum; j++) cout << ans[j] << " ";
        cout << endl;
    } else {
        cout << "NO" << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## Subarray Majority

### Solution 1: 

```py

```

## Irrational Path

### Solution 1: 

```py

```