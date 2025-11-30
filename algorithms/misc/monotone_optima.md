# Monotone Minima/Maxima

These are related to each other.  This is an example of a problem that can be solved with divide and conquer algorithm in O(nlogn) time complexity, and is much faster than the brute force O(n^2) solution.  

At the core for this problem you are looking at a monotone matrix.  Which in this context all you need to know is that it means that the row optima.  The row optima could be either minima or maxima.  But it needs to be montone with respect to the columns at which the row optima occurr.  In general, I believe the the row optima position is non-decreasing, so visualize it as increasing as it moves down and to the right.  This structure allows you to to perform a divide and conquer approach, because you only have to search the matrix in the top left portion and the bottom right for remaining searches.  Where you are dividing into two portions along the rows and taking the median row.  And you are also dividing into two portions along the columns, and that is based on where the row optima is at.  And you take the left segment for columns for the top segment of rows, and the right segment of columns for the bottom segment of rows. 

There may be some cases where things are in wrong order, and some reversing of arrays or sort is necessary.  I have not seen this yet. But I think it is possible.

Now let's take a specific example for the following

## Monotone Maxima

Let's try to solve monotone maxima, first what problem has this property.  If you are given two Arrays A and B.  And suppose B is concave downward, that means if you graph it, you get a peak. 

This fact that is concave downward, and I want to merge it with A by summing up these elements.  That is I'm trying to solve this zi = max(Ai + Bi), where i + j = k.  So this is a convolution, that if you brute force you see you get a matrix, and in fact it is a monotone matrix. 

You are creating a matrix with mat(i, j) with A[j] + B[i - j],  and this can be visualized as taking the reverse of array B and sliding it over A and summing up each time, and in this sense the peak is moving to the right of A at each row of the matrix. And sense the peak is moving through it works.  It is hard to understand but it is reasonable with this information. 

This code helps to visualize and see that it is also monotone matrix. 

```py
# A is random array
# B is concave downward sequence
n = 20
m = 20
A = [random.randint(-100, 100) for _ in range(n)]
diff = sorted([random.randint(-100, 100) for _ in range(m)], reverse = True)
B = [0] * m
B[0] = random.randint(-100, 100)
for i in range(1, m):
    B[i] = B[i - 1] + diff[i - 1]
def f(i, j):
    if j > i or i - j >= m: return -math.inf
    return A[j] + B[i - j]
def display(mat):
    for row in mat:
        print(",".join(map(str, row)))
print("A", A)
print("B", B)
# image of the concave downward sequence B
plt.scatter(range(m), B)
plt.show()
R = n + m - 1
C = n
mat = [[0] * C for _ in range(R)]
for i, j in product(range(R), range(C)):
    mat[i][j] = f(i, j)
row_maxima = [0] * R
for i in range(R):
    maxj = n 
    mx = -math.inf
    for j in range(C):
        if mat[i][j] > mx:
            mx = mat[i][j]
            maxj = j
    row_maxima[i] = maxj
display(mat)
print(row_maxima)
for i in range(R):
    print(f"====================={i}=====================")
    print("row maxima", row_maxima[i])
    for j in range(C):
        if i >= j and i - j < m:
            print("A", A[j], "B", B[i - j], "sum", A[j] + B[i - j], "mat", mat[i][j])
```

## Example of it applied to problem in Python

```py

```

## Example of it applied to problem in C++

```cpp
constexpr ll inf = 1ll << 60;

int n;
vector<pair<ll, ll>> x; // x : (a, b)

void read() {
    cin >> n;
    x.resize(n);
    for (auto &[a, b] : x) {
        cin >> a >> b;
    }
    sort(x.begin(), x.end(), [&](const auto &a, const auto &b) {
        return a.second < b.second;
    });
}

template <class F>
vector<ll> monotone_maxima(F &f, int h, int w) {
    vector<ll> ret(h);
    auto sol = [&](auto &&self, const int l_i, const int r_i, const int l_j, const int r_j) -> void {
        const int m_i = (l_i + r_i) / 2;
        int max_j = l_j;
        ll max_val = -inf;
        for (int j = l_j; j <= r_j; ++j) {
            const ll v = f(m_i, j);
            if (v > max_val) {
                max_j = j;
                max_val = v;
            }
        }
        ret[m_i] = max_val;

        if (l_i <= m_i - 1) {
            self(self, l_i, m_i - 1, l_j, max_j);
        }
        if (m_i + 1 <= r_i) {
            self(self, m_i + 1, r_i, max_j, r_j);
        }
    };
    sol(sol, 0, h - 1, 0, w - 1);
    return ret;
}

/*
what is a and b in this array? 
a is a vector of values for the right interval
b is a vector of values for the left interval

monotone_maxima
*/
vector<ll> max_plus_convolution(const vector<ll> &a, const vector<ll> &b) {
    const int n = (int)a.size(), m = (int)b.size();
    auto f = [&](int i, int j) {
        if (i < j or i - j >= m) {
            return -inf;
        }
        return a[j] + b[i - j];
    };

    return monotone_maxima(f, n + m - 1, n);
}

vector<ll> sol(const int l, const int r) {
    if (r - l == 1) {
        const vector<ll> ret = {-inf, x[l].first - x[l].second};
        return ret;
    }
    const int m = (l + r) / 2;
    const auto res_l = sol(l, m);
    const auto res_r = sol(m, r);

    vector<ll> sorted_l(m - l);
    for (int i = l; i < m; ++i) {
        sorted_l[i - l] = x[i].first;
    }
    sort(sorted_l.begin(), sorted_l.end(), greater());
    for (int i = 1; i < m - l; ++i) {
        sorted_l[i] += sorted_l[i - 1];
    }
    sorted_l.insert(sorted_l.begin(), -inf);
    // O(n)
    auto res = max_plus_convolution(res_r, sorted_l);

    for (int i = 0; i < (int)res_l.size(); ++i) {
        res[i] = max(res[i], res_l[i]);
    }
    for (int i = 0; i < (int)res_r.size(); ++i) {
        res[i] = max(res[i], res_r[i]);
    }
    return res;
}

void process() {
    auto ans = sol(0, n);
    for (int i = 1; i <= n; ++i) {
        cout << ans[i] << endl;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    read();
    process();
}
```