# Manhattan Trick

There are a few useful tricks with manhattan distances that from searching online is apparently called manhattan trick

The first one is this
max(|x2-x1|,|y2-y1|) = 2*(|s2-s1| + |t2-t1|)
where s = x + y, t = x - y
This one allows you to calculate the max value using the manhattan distance, here is an example of it being used in a grid, with binary search to calculate the distance to all the values in the s, t reference frame.  Which is just calculated by the distance from some starting s and starting t to the rest of the s and t values. 

The binary search logic is very important, read it carefully.

```cpp
int R, C;
vector<vector<int>> grid;
vector<int> psumx[2], psumy[2], arrx[2], arry[2];

int calc(const vector<int>& arr, const vector<int>& psum, int v) {
    int N = arr.size();
    int lo = 0, hi = N;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] <= v) lo = mid + 1;
        else hi = mid;
    }
    lo--;
    int lsum = 0;
    if (lo >= 0) lsum += v * (lo + 1) - psum[lo];
    int rsum = -v * (N - 1 - lo);
    if (N - 1 >= 0) rsum += psum[N - 1];
    if (lo >= 0) rsum -= psum[lo];
    int res = lsum + rsum;
    return res;
}

void solve() {
    cin >> R >> C;
    grid.assign(R, vector<int>(C, 0));
    for (int i = 0; i < 2; i++) {
        arrx[i].clear();
        arry[i].clear();
        psumx[i].clear();
        psumy[i].clear();
    }
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            cin >> grid[i][j];
            if (grid[i][j] == 1) {
                arrx[0].push_back(i + j);
                arry[0].push_back(i - j);
            } else if (grid[i][j] == 2) {
                arrx[1].push_back(i + j);
                arry[1].push_back(i - j);
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        sort(arrx[i].begin(), arrx[i].end());
        sort(arry[i].begin(), arry[i].end());
        psumx[i].assign(arrx[i].size(), 0);
        psumy[i].assign(arry[i].size(), 0);
        for (int j = 0; j < arrx[i].size(); j++) {
            psumx[i][j] = arrx[i][j];
            if (j > 0) {
                psumx[i][j] += psumx[i][j - 1];
            }
            psumy[i][j] = arry[i][j];
            if (j > 0) {
                psumy[i][j] += psumy[i][j - 1];
            }
        }
    }
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            int s1 = calc(arrx[0], psumx[0], i + j) + calc(arry[0], psumy[0], i - j);
            int s2 = calc(arrx[1], psumx[1], i + j) + calc(arry[1], psumy[1], i - j);
            int ans = abs(s1 - s2) / 2; // I dont' get this line but it is needed easily observed
            cout << ans << " ";
        }
        cout << endl;
    }
}
```


And sometimes it goes the other way that is |x2-x1|+|y2-y1| = max(|s2-s1|,|t2-t1|)


This can be solved in O(n) time using this manhattan trick

example where it uses the smax - smin, and dmax - dmin to find the two index that lead to maximum manhattan distance, finds the two points that lead to maximum manhattan distance.  which is either going to be from |s2-s1| or |t2-t1|

```py
def max_manhattan_distance(points, remove = -1):
    smin = dmin = math.inf
    smax = dmax = -math.inf
    smax_i = smin_i = dmax_i = dmin_i = None
    for i, (x, y) in enumerate(points):
        if remove == i: continue
        s = x + y
        d = x - y
        if s > smax:
            smax = s
            smax_i = i
        if s < smin:
            smin = s
            smin_i = i
        if d > dmax:
            dmax = d
            dmax_i = i
        if d < dmin:
            dmin = d
            dmin_i = i
    return (smax_i, smin_i) if smax - smin >= dmax - dmin else (dmax_i, dmin_i)
```