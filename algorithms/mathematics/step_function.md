# Step Function

## Parameterized step function

This is a simple function that will generate step function outputs, with given step size and height.  But also allows for horizontal and vertical shifts or translations of the data.

```py
def floor(x, y):
    return x // y
# provide step_size >= 1, and height >= 1
def step_function(x, step_size, height, horizontal_shift = 0, vertical_shift = 0):
    return floor(x + horizontal_shift, step_size) * height + vertical_shift
```

## Search step functions

q(x) = floor(N / x) and c(x) = ceil(N / x) are decreasing step functions in x ≥ 1.

They jump only at values where the quotient changes. Between jumps they are constant. Those constant stretches are the plateaus.

The thing to know is you can enumerate the plateaus in O(√N) time.

So if you identify a step function you can utilize that to find the optimal answer very fast. 

```cpp
const int64 INF = 1e18;
int64 a1, h1, a2, h2;

int64 ceil(int64 x, int64 y) {
    return (x + y - 1) / y;
}

int64 traversePlateaus() {
    int64 ans = INF;
    int64 curAttack = a1;
    while (true) {
        int64 rounds = ceil(h2, curAttack); // step functon is here
        ans = min(ans, max(0LL, curAttack - a1) + max(0LL, rounds * a2 - h1 + 1));
        if (rounds == 1) break; // we can't do better than this
        curAttack = ceil(h2, rounds - 1);
    }
    return ans;
}

void solve() {
    cin >> a1 >> h1 >> a2 >> h2;
    int64 ans = traversePlateaus();
    cout << ans << endl;
}
```