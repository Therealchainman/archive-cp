# Bitwise Operations

## Counting Bits

### Solution 1:  

```cpp

```

## Maximum XOR Subarray

### Solution 1: bitwise trie, prefix xor

The idea is to iterate over the elements in the array, and calculate the prefix xor up to the current element. For each prefix xor, we will find the best possible xor with any previous prefix xor using a bitwise trie. The trie will allow us to efficiently find the maximum xor sum that can be formed with the current prefix xor.

This works because you can calculate the xor over a segment by taking pref(r) ^ pref(l - 1) for instance. to get [l, r] segment xor. 

```cpp
const int BITS = 30;
int N;
vector<int> A;

struct Node {
    int children[2];
    int last;
    void init() {
        memset(children, 0, sizeof(children));
        last = -1;
    }
};
struct Trie {
    vector<Node> trie;
    void init() {
        Node root;
        root.init();
        trie.emplace_back(root);
    }
    void add(int mask) {
        int cur = 0;
        for (int i = BITS - 1; i >= 0; i--) {
            int bit = (mask >> i) & 1;
            if (trie[cur].children[bit] == 0) {
                Node root;
                root.init();
                trie[cur].children[bit] = trie.size();
                trie.emplace_back(root);
            }
            cur = trie[cur].children[bit];
        }
    }
    int find(int val) {
        int cur = 0, ans = 0;
        for (int i = BITS - 1; i >= 0; i--) {
            int valBit = (val >> i) & 1;
            if (trie[cur].children[valBit ^ 1] != 0) { // best result with xor is always with opposite bit
                ans |= (1 << i);
                cur = trie[cur].children[valBit ^ 1];
            } else {
                cur = trie[cur].children[valBit];
            }
        }
        return ans;
    }
};

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    Trie trie;
    trie.init();
    trie.add(0);
    int ans = 0, pref = 0;
    for (int i = 0; i < N; ++i) {
        pref ^= A[i];
        int best = trie.find(pref);
        ans = max(ans, best);
        trie.add(pref);
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Maximum XOR Subset

### Solution 1:  

```cpp

```

## 

### Solution 1:  

```cpp

```

## 

### Solution 1:  

```cpp

```

## XOR Pyramid Peak

### Solution 1:  bitmask, enumerating submasks, xor addition

The following analysis is for all of the XOR Pyramid problems: 

### Analysis 1
Given a sequence A of length N, let this represent the 0th row or the bottom of the pyramid.  

### Generating Function
Generate the nth row of the pyramid, which will be of length N - n
$$F_n(i) = \bigoplus_{j=0}^n [\binom{n}{j} \text{mod}2] \cdot A_{i+j} \quad \forall \quad 0<=i<N-n$$
 This part $\binom{n}{j}$ calculates the number of paths this particular element will contribute to the result at index i.  Essentially we know that if the number of times it is xor is even it will cancel itself out.  So in the end only the elements in sequence A that have an odd number of paths, will contribute to the final xor sum for the value of element in the row. 

Another way to visualize this is you have these fixed sized (n + 1) sliding windows over sequence A to generate each element in the nth row. 

The generating function above represents that sliding window with mathematical notation. 

### Lucas Theorem
Lucas theorem allows us to know when the remainder is equal to 0 under modulo 2 for a binomial coefficient. 

But by Lucas’s theorem (or the simple fact that $\binom{m}{k}$  is odd iff every 1-bit of k is also a 1-bit of m).  Or in otherwords k is a submask/subset of the bit representation of m.

From Lucas Theorem we get the following result:

$$\binom{n}{k} \bmod 2 = \begin{cases} 1 & \text{if} \; k \subseteq n \\ 0 & \text{otherwise} \end{cases}$$

This means we just need to enumerate the submasks.  So now the generating function becomes

$$F_n(i)=\bigoplus_{s\subseteq n} A_{i+s}$$

So how to solve this generating function to generate the nth row in O(NlogN) time complexity. 

For this specific problem: 

We can use the generating function to solve for the (N-1)th row which is the topmost.  

$$F_{N-1}(0)=\bigoplus_{s \subseteq N - 1} A_s$$

You just need to enumerate all the submasks of N - 1 and calculate the sum or you could also just loop over every index of $A_i$ and check if i is a submask of N - 1.  Either way is O(N) time complexity.  

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int rowMask = N - 1;
    int ans = A[0];
    for (int submask = rowMask; submask > 0; submask = (submask - 1) & rowMask) {
        ans ^= A[submask];
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## XOR Pyramid Diagonal

### Solution 1:  subset zeta transform, sum over subsets dp, bitmask dp, bit by bit enumeration of submasks

The generating function if we use it here becomes the following 
$$F_n(0)=\bigoplus_{s \subseteq n} A_s \quad \forall \quad 0 \leq n \le N$$
So we want to calculate this for all of the rows, 
$$F_0(0), F_1(0), F_2(0), \dots ,F_{N-1}(0)$$
Because of Lucas theorem that got us, here this is basically a zeta transformation, and we want to enumerate all the submasks for each value from 0 to N - 1.  This can be solved with the SOS dp algorithm in O(NlogN) time complexity.  

```cpp
const int LOG = 18;
int N;
vector<int> dp;

void solve() {
    cin >> N;
    dp.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> dp[i];
    }
    for (int i = 0; i < LOG; ++i) { // iterate over bits
        for (int mask = 0; mask < N; ++mask) {
            if ((mask >> i) & 1) dp[mask] ^= dp[mask ^ (1 << i)];
        }
    }
    for (int i = 0; i < N; ++i) {
        cout << dp[i] << " ";
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## XOR Pyramid Row

### Solution 1:  bitmask dp, sliding window pattern, bit by bit enumeration of submasks

This problem is harder because now you need to calculate the generating function for some sequence.  You can't use same approach as above because that would be a bout O($N^2$) time complexity. 

Must use this somehow. 
$$F_n(i)=\bigoplus_{s\subseteq n} A_{i+s} \quad \forall \; 0 \leq i \le \; N - n$$
The easiest solution to imagine is iterating over all the submasks of n, O(n) time complexity. Then for each submask s, you basically take the array and shift it to the left by s units for each element in the bottom row. 

This algorithm is useful for bitwise sliding window pattern matching or dynamic programming on fixed bitmasks across an array. 
Just remember one sweep per 1 bit of n folding in that shift, it should become as natural as any other prefix-sum or prefix-xor trick. 

```cpp
const int LOG = 18;
int N, K;
vector<int> A;

void solve() {
    cin >> N >> K;
    int n = N - K; // from bottom row
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < LOG; ++i) {
        int shift = (1 << i);
        if ((n >> i) & 1) {
            for (int j = 0; j < N - shift; ++j) {
                A[j] ^= A[j + shift];
            }
        }
    }
    for (int i = 0; i < K; ++i) {
        cout << A[i] << " ";
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## SOS Bit Problem

### Solution 1:  sum over subsets, sum over supersets

You are answer three different statements.

First we are going to be performing sum over subsets and over supersets, so we define our function $f(S)$: the frequency array that represents how many times each mask S appears in the input.

Then all you have to do is calculate the sums over subsets and supersets:

1. T | S = T, this requires counting the number of submasks of mask T, so $F(T) = \sum_{S \subseteq T} f(S)$
2. T & S = T, this requires counting the number of supersets of mask T, so $F(T) = \sum_{S \supseteq T} f(S)$
3. T & S != 0, this one requires a clever trick, observe that T and S must be sharing a bit.  So if I imagined inverting T, that is flipping all the bits, then I'm looking for S which are subsets of T, because those are the ones that don't share any bits with the original T.  So we can use the calculation of number of submasks of mask ~T. 

```cpp
const int LOG = 20;
int N;
vector<int> A, subsets, supersets;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    int endMask = 1 << LOG;
    subsets.assign(endMask, 0);
    supersets.assign(endMask, 0);
    for (int i = 0; i < N; ++i) {
        ++subsets[A[i]];
        ++supersets[A[i]];
    }
    for (int i = 0; i < LOG; ++i) { // iterate over bits
        for (int mask = 0; mask < endMask; ++mask) { // iterate over all masks
            if ((mask >> i) & 1) {
                int nmask = mask ^ (1 << i);
                subsets[mask] += subsets[nmask]; // subset
                supersets[nmask] += supersets[mask]; // superset
            }
        }
    }
    for (int x : A) {
        cout << subsets[x] << " " << supersets[x] << " " << N - subsets[(endMask - 1) ^ x] << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## And Subset Count

### Solution 1: sum over supersets, superset mobius inversion, frequency array

SAND algorithm (Superset AND Decomposition)
Problem: For a given collection of bitmasks, determine for each mask T how many subsets of the collection have a bitwise AND that equals T. 

1. **Frequency array** $f(S)$: how many times each mask S appears in the input.
2. **Superset Zeta transform** to compute $F(T) = \sum_{S \supseteq T} f(S)$: the total number of input elements "covering" T.
3. **Translate at least T into an exponential count**: $G(T) = 2^{F(T)} - 1$ is the number of non-empty subsets composed only of the elements that all cover T, so there AND >= T.
4. Observe that G(T) can represent the following basically 
Define a new function $g(T)$ = The number of subsets with AND = T, then you can make the statement that $$G(T) = \sum_{S \supseteq T} g(S)$$
5. **Superset mobius inversion** to recover $g(T)$ from $G(T)$, so you can calculate the number of subsets with AND = T. $$g(T) = G(T) - \sum_{S \supset T} g(S)$$

```cpp
const int LOG = 18, MOD = 1e9 + 7;
int N;
vector<int> A;

void solve() {
    cin >> N;
    int endMask = 1 << LOG;
    vector<int> supersets(endMask, 0);
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        ++supersets[A[i]];
    }
    for (int i = 0; i < LOG; ++i) { // iterate over
        for (int mask = 0; mask < endMask; ++mask) { // iterate over all masks
            if ((mask >> i) & 1) supersets[mask ^ (1 << i)] += supersets[mask]; // superset
        }
    }
    vector<int> pow2(N + 1, 1);
    for (int i = 1; i <= N; ++i) {
        pow2[i] = (2LL * pow2[i - 1]) % MOD; // precompute powers of 2
    }
    vector<int> G(endMask, 0);
    for (int mask = 0; mask < endMask; ++mask) {
        G[mask] = (pow2[supersets[mask]] - 1 + MOD) % MOD;
    }
    vector<int> g = G; // copy for mobius inversion
    for (int i = 0; i < LOG; ++i) { // iterate over
        for (int mask = 0; mask < endMask; ++mask) { // iterate over all masks
            int nmask = mask ^ (1 << i);
            if ((mask >> i) & 1) g[nmask] = (g[nmask] - g[mask] + MOD) % MOD; // inverse superset
        }
    }
    for (int i = 0; i <= N; ++i) {
        cout << g[i] << " ";
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```