# Meta Hacker Cup 2015

# Round 2

## 

```py

```

## 

```py

```

## Problem C: Circular Circles

minimum spanning tree, undirected graph, multiset, red black binary search trees

```cpp
const int mod = 1e9 + 7;
const int mod2 = 1e9;
multiset<int> weight_pairs, inter_circle_weights;
vector<multiset<int>> top_circle_weights, bot_circle_weights;
vector<int> X, Y, I, W, CW;

int maximum(multiset<int>& ms) {
    auto max_val_iterator = ms.rbegin();
    return max_val_iterator != ms.rend() ? *max_val_iterator : 0;
}

void erase(multiset<int>& s, int x) {
	s.erase(s.find(x));
}

void solve() {
    // N circles
    // M nodes in each circle
    // N extra edges to connect all the circles into a super circle
    int N = read(), M = read(), E = read(), K = read();
    X.assign(N, 0);
    Y.assign(N, 0);
    I.assign(E, 0);
    W.assign(E, 0);
    for (int i = 0; i < K; i++) {
        X[i] = read();
    }
    int Ax = read(), Bx = read(), Cx = read();
    for (int i = K; i < N; i++) {
        X[i] = (Ax * X[i - 2] + Bx * X[i - 1] + Cx) % M;
    }
    for (int i = 0; i < K; i++) {
        Y[i] = read();
    }
    int Ay = read(), By = read(), Cy = read();
    for (int i = K; i < N; i++) {
        Y[i] = (Ay * Y[i - 2] + By * Y[i - 1] + Cy) % M;
    }
    for (int i = 0; i < K; i++) {
        I[i] = read();
    }
    int Ai = read(), Bi = read(), Ci = read();
    for (int i = K; i < E; i++) {
        I[i] = (Ai * I[i - 2] + Bi * I[i - 1] + Ci) % (N * M + N);
    }
    for (int i = 0; i < K; i++) {
        W[i] = read();
    }
    int Aw = read(), Bw = read(), Cw = read();
    for (int i = K; i < E; i++) {
        W[i] = (Aw * W[i - 2] + Bw * W[i - 1] + Cw) % mod2;
    }
    int res = 1;
    weight_pairs.clear();
    inter_circle_weights.clear();
    top_circle_weights.assign(N, multiset<int>());
    bot_circle_weights.assign(N, multiset<int>());
    // N * M circle edges
    for (int i = 0; i < N * M; i++) {
        int c = i / M;
        int e = i % M;
        if (e >= min(X[c], Y[c]) && e < max(X[c], Y[c])) {
            // TOP
            top_circle_weights[c].insert(1);
        } else {
            // BOT
            bot_circle_weights[c].insert(1);
        }
        if (e == 0) {
            weight_pairs.insert(X[c] == Y[c] ? 0 : 1);
        }
    }
    // N inter circle edges
    for (int i = 0; i < N; i++) {
        inter_circle_weights.insert(1);
    }
    CW.assign(N * M + N, 1);
    int total_weight = N * M + N;
    int circle_weight = N;
    for (int q = 0; q < E; q++) {
        int weight = W[q], edge = I[q];
        int cur_weight;
        // belongs to inter-circle edge
        if (edge >= N * M) {
            cur_weight = CW[edge];
            erase(inter_circle_weights, cur_weight);
            inter_circle_weights.insert(weight);
        } else {
            int c = edge / M;
            int e = edge % M;
            cur_weight = CW[edge];
            int top_max = maximum(top_circle_weights[c]), bot_max = maximum(bot_circle_weights[c]);
            int cur_pair_max = top_max + bot_max - max(top_max, bot_max);
            int cur_circle_max = max(top_max, bot_max);
            erase(weight_pairs, cur_pair_max);
            if (e >= min(X[c], Y[c]) && e < max(X[c], Y[c])) {
                // TOP
                erase(top_circle_weights[c], cur_weight);
                top_circle_weights[c].insert(weight);
            } else {
                // BOT
                erase(bot_circle_weights[c], cur_weight);
                bot_circle_weights[c].insert(weight);
            }
            top_max = maximum(top_circle_weights[c]), bot_max = maximum(bot_circle_weights[c]);
            int delta = max(top_max, bot_max) - cur_circle_max;
            circle_weight += delta;
            int pair_delta = top_max + bot_max - max(top_max, bot_max);
            weight_pairs.insert(pair_delta);
        }
        CW[edge] = weight;
        int max_extra = max(maximum(weight_pairs), maximum(inter_circle_weights));
        int max_weight_pairs = maximum(weight_pairs);
        int max_inter_circle_weights = maximum(inter_circle_weights);
        total_weight = total_weight + weight - cur_weight;
        int mst_weight = (total_weight - max_extra - circle_weight) % mod;
        res = (res * mst_weight) % mod;
    }
    cout << res;
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T = read();
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
        cout << endl;
    }
    return 0;
}
```

##

```py

```

##

```py

```

##

```py

```

##

```py

```