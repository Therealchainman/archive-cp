# Atcoder Beginner Contest 343

## D - Diversity of Scores 

### Solution 1:  frequency counter

```py
def main():
    N, T = map(int, input().split())
    freq = Counter({0: N})
    scores = [0] * N
    ans = 1
    for _ in range(T):
        a, b = map(int, input().split())
        a -= 1
        freq[scores[a]] -= 1
        if freq[scores[a]] == 0: ans -= 1
        scores[a] += b
        freq[scores[a]] += 1
        if freq[scores[a]] == 1: ans += 1
        print(ans)

if __name__ == '__main__':
    main()
```

## E - 7x7x7 

### Solution 1:  fix cube at origin, iterate over all possible configs for the other two cubes, check if the volume is correct, range intersection

```py
def intersection(*ranges):
    return max(0, min([e for _, e in ranges]) - max([s for s, _ in ranges]))

def output(x1, x2, y1, y2, z1, z2):
    res = f"0 0 0 {x1} {y1} {z1} {x2} {y2} {z2}"
    return res

VOLUME = 7 * 7 * 7 * 3
def main():
    v1, v2, v3 = map(int, input().split())
    if v1 + 2 * v2 + 3 * v3 != VOLUME: return print("No")
    def check(x1, x2, y1, y2, z1, z2):
        x_ranges = [(0, 7), (x1, x1 + 7), (x2, x2 + 7)]
        y_ranges = [(0, 7), (y1, y1 + 7), (y2, y2 + 7)]
        z_ranges = [(0, 7), (z1, z1 + 7), (z2, z2 + 7)]
        v3_x = intersection(*x_ranges)
        v3_y = intersection(*y_ranges)
        v3_z = intersection(*z_ranges)
        vol3 = v3_x * v3_y * v3_z
        if vol3 != v3: return False
        vol2 = 0
        for i in range(3):
            for j in range(i + 1, 3):
                v2_x = intersection(x_ranges[i], x_ranges[j])
                v2_y = intersection(y_ranges[i], y_ranges[j])
                v2_z = intersection(z_ranges[i], z_ranges[j])
                vol2 += v2_x * v2_y * v2_z
        vol2 -= 3 * vol3
        return vol2 == v2
    L = 7
    for x1 in range(-L, L + 1):
        for x2 in range(-L, L + 1):
            for y1 in range(-L, L + 1):
                for y2 in range(-L, L + 1):
                    for z1 in range(-L, L + 1):
                        for z2 in range(-L, L + 1):
                            if check(x1, x2, y1, y2, z1, z2):
                                print("Yes")
                                print(output(x1, x2, y1, y2, z1, z2))
                                return
    print("No")

if __name__ == '__main__':
    main()
```

```cpp
const int VOLUME = 7 * 7 * 7 * 3, L = 7;
int v1, v2, v3;

// Function to find the intersection length of the ranges
int intersection(const vector<pair<int, int>>& ranges) {
    int start_max = INT_MIN;
    int end_min = INT_MAX;
    for (auto& range : ranges) {
        start_max = max(start_max, range.first);
        end_min = min(end_min, range.second);
    }
    return max(0LL, end_min - start_max);
}

bool check(int x1, int x2, int y1, int y2, int z1, int z2) {
    vector<pair<int, int>> x_ranges = {{0, 7}, {x1, x1 + 7}, {x2, x2 + 7}};
    vector<pair<int, int>> y_ranges = {{0, 7}, {y1, y1 + 7}, {y2, y2 + 7}};
    vector<pair<int, int>> z_ranges = {{0, 7}, {z1, z1 + 7}, {z2, z2 + 7}};
    int v3_x = intersection(x_ranges);
    int v3_y = intersection(y_ranges);
    int v3_z = intersection(z_ranges);
    int vol3 = v3_x * v3_y * v3_z;
    if (vol3 != v3) return false;
    int vol2 = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            vector<pair<int, int>> temp_x = {x_ranges[i], x_ranges[j]};
            vector<pair<int, int>> temp_y = {y_ranges[i], y_ranges[j]};
            vector<pair<int, int>> temp_z = {z_ranges[i], z_ranges[j]};
            
            int v2_x = intersection(temp_x);
            int v2_y = intersection(temp_y);
            int v2_z = intersection(temp_z);
            
            vol2 += v2_x * v2_y * v2_z;
        }
    }
    vol2 -= 3 * vol3;
    return vol2 == v2;
}

signed main() {
    cin >> v1 >> v2 >> v3;
    if (v1 + 2 * v2 + 3 * v3 != VOLUME) {
        cout << "No" << endl;
        return 0;
    }
    for (int x1 = -L; x1 <= L; ++x1) {
        for (int x2 = -L; x2 <= L; ++x2) {
            for (int y1 = -L; y1 <= L; ++y1) {
                for (int y2 = -L; y2 <= L; ++y2) {
                    for (int z1 = -L; z1 <= L; ++z1) {
                        for (int z2 = -L; z2 <= L; ++z2) {
                            if (check(x1, x2, y1, y2, z1, z2)) {
                                cout << "Yes" << endl;
                                cout << 0 << " " << 0 << " " << 0 << " " << x1 << " " << y1 << " " << z1 << " " << x2 << " " << y2 << " " << z2 << endl;
                                return 0;
                            }
                        }
                    }
                }
            }
        }
    }
    cout << "No" << endl;
    return 0;
}
```

## F - Second Largest Query 

### Solution 1:  segment tree, store max1, max2, cnt1, cnt2 in each segment

```py
class SegmentTree:
    def __init__(self, n, neutral, initial_arr):
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.first = [neutral for _ in range(self.size * 2)]
        self.second = [neutral for _ in range(self.size * 2)]
        self.build(initial_arr)

    def build(self, initial_arr):
        for i, segment_idx in enumerate(range(self.n)):
            segment_idx += self.size - 1
            val = initial_arr[i]
            self.first[segment_idx] = [val, 1]
            self.ascend(segment_idx)

    def ascend(self, segment_idx):
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            pairs = [self.first[left_segment_idx], self.first[right_segment_idx], self.second[left_segment_idx], self.second[right_segment_idx]]
            max1 = max2 = cnt1 = cnt2 = 0
            for v, _ in pairs:
                if v > max1:
                    max2 = max1 
                    max1 = v
                elif max2 < v < max1:
                    max2 = v
            for v, c in pairs:
                if max1 == v:
                    cnt1 += c
                elif max2 == v:
                    cnt2 += c
            self.first[segment_idx] = [max1, cnt1]
            self.second[segment_idx] = [max2, cnt2]
        
    def update(self, segment_idx, val):
        segment_idx += self.size - 1
        self.first[segment_idx] = [val, 1]
        self.ascend(segment_idx)
            
    def query(self, left, right):
        stack = [(0, self.size, 0)]
        max1 = max2 = cnt1 = cnt2 = 0
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                values = [self.first[segment_idx], self.second[segment_idx]]
                for v, _ in values:
                    if v > max1:
                        max2 = max1
                        cnt2 = cnt1
                        max1 = v
                        cnt1 = 0
                    elif max2 < v < max1:
                        max2 = v
                        cnt2 = 0
                for v, c in values:
                    if max1 == v:
                        cnt1 += c
                    elif max2 == v:
                        cnt2 += c
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return cnt2
    
    def __repr__(self):
        return f"first array: {self.first}, second array: {self.second}"
def main():
    N, Q = map(int, input().split())
    arr = list(map(int, input().split()))
    seg = SegmentTree(N, [0, 0], arr)
    for _ in range(Q):
        query = list(map(int, input().split()))
        if query[0] == 1: # point update 
            p, x = query[1:]
            p -= 1
            seg.update(p, x)
        else: # range query
            l, r = query[1:]
            l -= 1
            print(seg.query(l, r))

if __name__ == '__main__':
    main()
```

## G - Compress Strings 

### Solution 1: 

```py

```

