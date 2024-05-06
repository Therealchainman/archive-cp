# Codeforces Educational 165 Div 2

## D. Shap Game

### Solution 1:  greedy, max heap, alice pick subset, bob pick to minimize (always largest b cost)

```py
from heapq import heappush, heappop

def main():
    n, k = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    arr = sorted([(a, b) for a, b in zip(A, B)], key = lambda x: x[1])
    max_heap = []
    cost = ans = 0
    for i in range(1, k + 1):
        a, b = arr[-i]
        cost += a
        heappush(max_heap, -a)
    psum = [0] * n
    for i in range(n):
        a, b = arr[i]
        psum[i] = max(0, b - a)
        if i > 0: psum[i] += psum[i - 1]
    for i in range(n - k - 1, -1, -1):
        a, b = arr[i]
        ans = max(ans, psum[i] - cost)
        heappush(max_heap, -a)
        cost += a
        cost += heappop(max_heap)
    print(ans)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## 

### Solution 1: 

```py

```

## F. Card Pairing

### Solution 1:  xor hashing, dynamic programming, bit manipulation 

```cpp
mt19937_64 rnd(12341234);

const int INF = 1e9, MAXN = 1e3 + 5;
int N, K, h;
int deck[MAXN], freq[MAXN], vals[MAXN], phs[MAXN];
vector<int> dp;
bool flag;

// finds the next state from pos where the prefix hash is equal to h
int get_state(int pos, int h) {
    for (int i = pos; i < N; i += 2) {
        if (phs[i] == h) return i;
    }
    return -1;
}

vector<int> suffix_parity(int pos) {
    vector<int> res(K, 0);
    for (int i = pos; i < N; i++) {
        res[deck[i]] ^= 1;
    }
    return res;
}

void solve() {
    cin >> N >> K;
    memset(freq, 0, sizeof(freq));
    dp.assign(N, INF);
    for (int i = 0; i < N; i++) {
        cin >> deck[i];
        deck[i]--;
        freq[deck[i]]++;
    }
    int max_score = 0;
    h = 0;
    for (int i = 0; i < K; i++) {
        vals[i] = rnd();
        max_score += freq[i] / 2;
    }
    for (int i = 0; i < N; i++) {
        h ^= vals[deck[i]];
        phs[i] = h;
    }
    h = 0;
    for (int i = 0; i < K; i++) h ^= vals[i];
    int pos = get_state(K - 1, h);
    if (pos == -1) {
        cout << max_score << endl;
        return;
    }
    int ans = INF;
    dp[pos] = 0;
    for (int k = K - 1; k < N; k += 2) {
        vector<int> spar = suffix_parity(k + 1), even, odd;
        for (int i = 0; i < K; i++) {
            if (spar[i]) {
                odd.push_back(i);
            } else {
                even.push_back(i);
            }
        }
        int es = even.size(), os = odd.size();
        flag = true;
        for (int i = 0; i < os && flag; i++) {
            for (int j = 0; j < i && flag; j++) {
                int x = odd[i], y = odd[j];
                int add = 2;
                int hlook = phs[k] ^ vals[x] ^ vals[y];
                int pos = get_state(k, hlook);
                if (pos == -1) {
                    flag = false;
                    ans = min(ans, dp[k] + add);
                } else {
                    dp[pos] = min(dp[pos], dp[k] + add);
                }
            }
        }
        flag = true;
        for (int i = 0; i < es && flag; i++) {
            for (int j = 0; j < i && flag; j++) {
                int x = even[i], y = even[j];
                int add = 0;
                int hlook = phs[k] ^ vals[x] ^ vals[y];
                int pos = get_state(k, hlook);
                if (pos == -1) {
                    flag = false;
                    ans = min(ans, dp[k] + add);
                } else {
                    dp[pos] = min(dp[pos], dp[k] + add);
                }
            }
        }
        flag = true;
        for (int i = 0; i < os && flag; i++) {
            for (int j = 0; j < es && flag; j++) {
                int x = odd[i], y = even[j];
                int add = 1;
                int hlook = phs[k] ^ vals[x] ^ vals[y];
                int pos = get_state(k, hlook);
                if (pos == -1) {
                    flag = false;
                    ans = min(ans, dp[k] + add);
                } else {
                    dp[pos] = min(dp[pos], dp[k] + add);
                }
            }
        }
    }
    cout << max_score - ans << endl;
    return;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```