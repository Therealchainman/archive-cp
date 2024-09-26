# Meta Hacker Cup 2024

## Practice Round

## Problem A: Walk the Line

### Solution 1: greedy

1. Always take the fastest person to help each other person across the line.

```cpp
// string name = "walk_the_line_sample_input.txt";
// string name = "walk_the_line_validation_input.txt";
string name = "walk_the_line_input.txt";
const int INF = 1e18;
int N, K, x;

string solve() {
    int best = INF;
    cin >> N >> K;
    for (int i = 0; i < N; i++) {
        cin >> x;
        best = min(best, x);
    }
    int ans = max(best, (2 * N - 3) * best);
    if (ans <= K) return "YES";
    return "NO";
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": " << solve() << endl;
    }
    return 0;
}
```

## Problem B: Line by Line

### Solution 1:  math, probability

1. Calculate the probability of having N - 1 correct lines, need to convert into decimal form and not percentage so that it is correct.
2. Then that is the target probability with N lines, so just take the Nth root to get what is the necessary probability to multiple to it.  Cause you have to multiple by this value N times to get the target probability. 
3. Then convert back to percentage and subtract the original percentage to get the difference.

```cpp
// string name = "line_by_line_sample_input.txt";
// string name = "line_by_line_validation_input.txt";
string name = "line_by_line_input.txt";
const int INF = 1e18;
int N, P;

long double solve() {
    cin >> N >> P;
    long double p = P / 100.0;
    long double prob1 = pow(p, N - 1);
    long double ans = pow(prob1, 1 / (long double)N) * 100.0;
    return ans - P;
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": " << fixed << setprecision(15) << solve() << endl;
    }
    return 0;
}
```

## Problem C: Fall in Line

### Solution 1: 

1. It was probability problem. 
1. If there were more than N / 2 points along the optimal path, it turns out the probability of picking two points at random and having them be on the optimal path is less than 1/4.  So the probability of picking two points no on optimal path is greater than 3/4.  So if you keep picking two points (3/4)^K, if you pick K times this is the probability that it fails K times.  The probability will get very low because the number is less than 1.  And therefore you will be guaranteed to have picked two points along the optimal path. 
1. If the optimal path is under half it doesn't really matter you can just return N.  

```cpp
string base = "fall_in_line";
// string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
string name = base + "_input.txt";

random_device rd;
mt19937_64 gen(rd());
int randint(int l, int r) {
    uniform_int_distribution<int> dist(l, r);
    return dist(gen);
}

const int INF = 1e9;
int N;
vector<pair<int, int>> points;

int outer_product(const pair<int, int>& v1, const pair<int, int>& v2) {
    return v1.x * v2.y - v1.y * v2.x;
}

// calculate number of points non-collinear with points p1 and p2.  
int calc(const pair<int, int>& p1, const pair<int, int>& p2) {
    int ans = 0;
    for (int i = 0; i < N; i++) {
        pair<int, int> v1 = {points[i].x - p1.x, points[i].y - p1.y};
        pair<int, int> v2 = {points[i].x - p2.x, points[i].y - p2.y};
        if (outer_product(v1, v2) != 0) ans++;
    }
    return ans;
}

void solve() {
    cin >> N;
    points.resize(N);
    for (int i = 0; i < N; i++) {
        int x, y;
        cin >> x >> y;
        points[i] = {x, y};
    }
    int ans = INF;
    for (int i = 0; i < 100; i++) {
        int u = 0, v = 0;
        while (u == v) {
            u = randint(0, N - 1);
            v = randint(0, N - 1);
        }
        ans = min(ans, calc(points[u], points[v]));
    }
    cout << ans << endl;
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
    }
    return 0;
}
```

## Problem D1: Line of Delivery (Part 1)

### Solution 1:  fenwick tree, find the closest number to G and the distance to it

1. I didn't realize it at first but all the energies are unique, so basically the collisions are very simple to model
2. So it is easy to find the closest it gets to the target from the left of the goal and the right of the goal.
3. One of these is the answer, or both of them are.  But how do you find the index of the stone that is the closest to the goal?
4. I observed that the index of the stones is sorted, that is the leftmost stone is the last thrown and the rightmost stone is the first thrown.
5. So really if you just know how many stones are to at the given position, you know it is the N - count of stones at that position. 
6. For example if you have N = 10, and count of stones = 3, that means it is 3rd from the last thrown stones, so that would be 8th stone thrown that reaches that position. 

```cpp
string base = "line_of_delivery_part_1";
// string name = base + "_sample_input.txt";
// string name = base + "_validation_input.txt";
string name = base + "_input.txt";
const int INF = 1e18, MAXN = 1e6 + 5;
int N, G;
struct FenwickTree {
    vector<int> nodes;
    int neutral = 0;

    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, int val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : 0;
    }

    int query(int idx) {
        int result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};

FenwickTree ft;

void solve() {
    cin >> N >> G;
    int left_g = -INF, right_g = INF;
    unordered_multiset<int> nums;
    for (int i = 0; i < N; i++) {
        int x;
        cin >> x;
        nums.insert(x);
        if (x < G) {
            left_g = max(left_g, x);
        } else {
            right_g = min(right_g, x);
        }
        ft.update(x, 1);
    }
    int ans = INF, mini = INF;
    if (left_g != -INF) {
        ans = min(ans, G - left_g);
    }
    if (right_g != INF) {
        ans = min(ans, right_g - G);
    }
    if (G - left_g == ans) {
        int idx = N - ft.query(left_g);
        mini = min(mini, idx);
    }
    if (right_g - G == ans) {
        int idx = N - ft.query(right_g);
        mini = min(mini, idx);
    }
    cout << mini + 1 << " " << ans << endl;
    for (int x : nums) {
        ft.update(x, -1);
    }
}

signed main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    ft.init(MAXN);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        cout << "Case #" << i << ": ";
        solve();
    }
    return 0;
}
```

## Problem D2: Line of Delivery (Part 2)

### Solution 1: 

1. This one looks more challenging because now the stones have unit width, that is they take up spots, and you can have duplicate energies now.  This time a stone will stop at position just in front of the next it collides with and transfers it's energy. 
2. It is a little more complex than part 1
now if you have 2 balls before position 9, and you throw a ball with strength 9, it will send the last ball to position 11, because the balls take up space.  So that means it will go 9 + # of balls before it.
But it is complicated because what of the other balls between.  

I get feeling it may involve stack. 

```cpp

```

## Round 1

##

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

##

### Solution 1: 

```cpp

```
