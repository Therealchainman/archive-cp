# Additional Problems

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## Intersection Points

### Solution 1:  PURQ, point updates, range queries, sum, BIT, fenwick, line sweep, 2D grid to 1D range query, coordinate compression

```cpp
int N;

int neutral = 0;
struct FenwickTree {
    vector<int> nodes;
    
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
        return query(right) - query(left - 1);
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

struct Vertical {
    int x, y1, y2;
};

struct Horizontal {
    int y, x1, x2;
};

signed main() {
	int x1, y1, x2, y2;
    cin >> N;
    vector<Vertical> verticals;
    vector<Horizontal> horizontals;
    vector<int> values;
    for (int i = 0; i < N; i++) {
        cin >> x1 >> y1 >> x2 >> y2;
        values.push_back(x1);
        values.push_back(x2);
        values.push_back(y1);
        values.push_back(y2);
        if (x1 == x2) {
            verticals.push_back({x1, y1, y2});
        } else {
            horizontals.push_back({y1, x1, x2});
        }
    }
    FenwickTree fenwick;
    fenwick.init(values.size());
    // COORDINATE COMPRESSION
    sort(values.begin(), values.end());
    auto it = unique(values.begin(), values.end()); 
    values.resize(distance(values.begin(), it));
    map<int, int> compress;
    for (int &v : values) {
        compress[v] = compress.size() + 1;
    }
    // 1D LINE SWEEP ALGORITHM
    vector<tuple<int, int, int>> events;
    for (int i = 0; i < (int)horizontals.size(); i++) {
        events.emplace_back(compress[horizontals[i].x1], 1, i);
        events.emplace_back(compress[horizontals[i].x2], -1, i);
    }
    for (int i = 0; i < (int)verticals.size(); i++) {
        events.emplace_back(compress[verticals[i].x], 0, i);
    }
    sort(events.begin(), events.end());
    int ans = 0, l, r;
    for (auto &[_, t, idx] : events) {
        int y = compress[horizontals[idx].y];
        if (t == -1) {
            fenwick.update(y, -1);
        } else if (t == 0) {
            l = compress[verticals[idx].y1], r = compress[verticals[idx].y2];
            ans += fenwick.query(l, r);
        } else {
            fenwick.update(y, 1);
        }
    }
    cout << ans << endl;
}
```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```