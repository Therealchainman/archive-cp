# Sparse Tables

sparse tables are best for range queries with an immutable array or static array.  They even have O(1) query for indempotent functions and the rest O(logn)

Precompute minima of intervals whose lengths are powers of two starting at every index (sparse table).

- idempotent
- associative

## Idempotent Functions

Idempotent functions are really useful for sparse tables because they allow O(1) operations.
Idempotent function means you can apply the function multiple times and it will not change the result.  For example, min, max, gcd, lcm, etc.  But not sum, product, etc.
f(x,x) = x is condition for idempotent function
sum(x,x) = 2x that is why sum function is not idempotent.
Because you don't care about applying function multiple times you can apply it over an overlapping range in the query, and you can cover any power of two lengthed segment
by using two power of two lengthed segments.


```cpp
// Requirements for Op::combine:
// - associative: combine(a, combine(b,c)) == combine(combine(a,b), c)
// - idempotent: combine(x, x) == x
// Sparse table queries rely on idempotence to safely overlap intervals.

template <class T, class Op>
struct SparseTable {
    vector<int> lg;          // floor(log2(i))
    vector<vector<T>> st;    // st[k][i] = combine over [i, i + 2^k)

    SparseTable() = default;
    explicit SparseTable(const vector<T>& a) { build(a); }

    void build(const vector<T>& a) {
        int n = a.size();
        lg.assign(n + 1, 0);
        for (int i = 2; i <= n; ++i) lg[i] = lg[i / 2] + 1;

        int K = (n == 0) ? 0 : (lg[n] + 1);
        st.assign(K, vector<T>(n));
        if (n == 0) return;

        st[0] = a;
        for (int k = 1; k < K; ++k) {
            int len = 1 << k;
            int half = len >> 1;
            for (int i = 0; i + len <= n; ++i) {
                st[k][i] = Op::combine(st[k - 1][i], st[k - 1][i + half]);
            }
        }
    }

    // Query on half-closed interval [l, r]
    // Precondition: 0 <= l <= r <= n
    T query(int l, int r) const {
        int len = r - l + 1;
        int k = lg[len];
        // Two blocks cover [l,r], potentially overlapping: OK only if idempotent.
        return Op::combine(st[k][l], st[k][r - (1 << k) + 1]);
    }
};

// ----- Plug-in ops (examples) -----

template <class T>
struct MinOp {
    static T identity() {
        return numeric_limits<T>::max();
    }

    static T combine(T a, T b) {
        return min(a, b);
    }
};
template <class T>
struct MaxOp {
    static T identity() {
        return numeric_limits<T>::lowest();
    }

    static T combine(T a, T b) {
        return max(a, b);
    }
};

template <class T>
struct GcdOp {
    static T identity() {
        return T(0);
    }

    static T combine(T a, T b) {
        return gcd(a, b);
    }
};

template <class T>
struct BitOrOp {
    static T identity() {
        return T(0);
    }

    static T combine(T a, T b) {
        return a | b;
    }
};

template <class T>
struct BitAndOp {
    static T identity() {
        return ~T(0);
    }

    static T combine(T a, T b) {
        return a & b;
    }
};
```

## Disjoint Sparse Table

A Disjoint Sparse Table is a data structure designed for answering range queries on a static array. Its key feature is the ability to handle any associative operation in 𝑂(1)
 query time after an 𝑂(𝑁log𝑁)
 preprocessing step. This makes it more versatile than a standard Sparse Table, which is limited to idempotent operations.

The primary limitation of a standard Sparse Table is its reliance on overlapping query ranges. This works for idempotent operations like min
 or max
 (e.g., min(𝑎,min(𝑏,𝑐))=min(𝑎,𝑏,𝑐)
), but fails for non-idempotent operations like summation, where including an element twice would be incorrect.

The Disjoint Sparse Table overcomes this by ensuring that the precomputed blocks used to answer any query are always disjoint (non-overlapping). This guarantees each element in the query range is considered exactly once. It achieves this by precomputing answers for ranges that grow outwards from the center of power-of-two-sized blocks.

When to Use It

You need to answer range queries on a static array.
The query operation is associative but not idempotent (e.g., sum, product, subtraction, XOR).
Query time must be strictly 𝑂(1).

