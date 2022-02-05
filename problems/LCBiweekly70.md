# Leetcode Biweekly Contest 70

## 2144. Minimum Cost of Buying Candies With Discount

### Solution: modular math + array iteration + sort

```c++
int minimumCost(vector<int>& cost) {
    sort(cost.begin(),cost.end());
    int n = cost.size(), sumCost = 0;
    for (int i = 0;i<n;i++) {
        sumCost += (i%3==n%3 ? 0 : cost[i]);
    }
    return sumCost;
}
```

## 2145. Count the Hidden Sequences

### Solution: math bounds

```c++
int numberOfArrays(vector<int>& D, int lower, int upper) {
    long long mn = 0, mx = 0, num = 0;
    for (int i=0;i<D.size();i++) {
        num+=D[i];
        mn = min(mn,num);
        mx = max(mx,num);
    }
    
    mx += (lower-mn);
    return max(0LL,(long long)upper-mx+1LL);
}
```

## 2146. K Highest Ranked Items Within a Price Range

### Solution: BFS + custom sort

TC: O(mnlog(mn))

```c++
struct Item {
    int row, col, price, dist;
    void init(int r, int c, int p, int d) {
        row = r, col = c, price = p, dist = d;
    }
};
class Solution {
public:
    vector<vector<int>> highestRankedKItems(vector<vector<int>>& grid, vector<int>& pricing, vector<int>& start, int k) {
        vector<Item> items;
        int R = grid.size(), C = grid[0].size(), low = pricing[0], high = pricing[1];
        queue<Item> q;
        Item sitem;
        sitem.init(start[0],start[1],grid[start[0]][start[1]],0);
        auto inPrice = [&](const int& i, const int& j) {
            return grid[i][j]>=low && grid[i][j]<=high;
        };
        if (inPrice(start[0],start[1])) {
            items.push_back(sitem);
        }
        grid[start[0]][start[1]]=-1;
        q.push(sitem);
        auto inBounds = [&](const int& i, const int& j) {
            return i>=0 && i<R && j>=0 &&j<C;
        };
        while (!q.empty()) {
            Item curItem = q.front();
            q.pop();
            for (int dr = -1;dr<=1;dr++) {
                for (int dc =-1;dc<=1;dc++) {
                    if (abs(dc+dr)!=1) continue;
                    int nr = curItem.row+dr, nc = curItem.col+dc;
                    if (!inBounds(nr,nc) || grid[nr][nc]==-1) continue;
                    if (grid[nr][nc]>0) {
                        Item item;
                        item.init(nr,nc,grid[nr][nc],curItem.dist+1);
                        q.push(item);
                        if (inPrice(nr,nc)) {
                            items.push_back(item);
                        }
                    }
                    grid[nr][nc]=-1; // set as visited
                }
            }
        }
        sort(items.begin(),items.end(),[&](const Item& a, const Item& b) {
            if (a.dist != b.dist) return a.dist < b.dist;
            if (a.price != b.price) return a.price < b.price;
            if (a.row != b.row) return a.row < b.row;
            return a.col < b.col;
        });
        vector<vector<int>> res;
        for (int i = 0;i<k && i<items.size();i++) {
            res.push_back({items[i].row,items[i].col});
        }
        return res;
    }
```

### Solution: BFS + priority queue + custom sort

TC: O(mnlog(k))

```c++
struct Item {
    int row, col, price, dist;
    void init(int r, int c, int p, int d) {
        row = r, col = c, price = p, dist = d;
    }
};
struct compare {
    bool operator()(const Item& a, const Item& b) {
        if (a.dist != b.dist) return a.dist < b.dist;
        if (a.price != b.price) return a.price < b.price;
        if (a.row != b.row) return a.row < b.row;
        return a.col < b.col;
    }  
};
class Solution {
public:
    vector<vector<int>> highestRankedKItems(vector<vector<int>>& grid, vector<int>& pricing, vector<int>& start, int k) {
        priority_queue<Item,vector<Item>,compare> heap;
        int R = grid.size(), C = grid[0].size(), low = pricing[0], high = pricing[1];
        queue<Item> q;
        Item sitem;
        sitem.init(start[0],start[1],grid[start[0]][start[1]],0);
        auto inPrice = [&](const int& i, const int& j) {
            return grid[i][j]>=low && grid[i][j]<=high;
        };
        if (inPrice(start[0],start[1])) {
            heap.push(sitem);
        }
        grid[start[0]][start[1]]=-1;
        q.push(sitem);
        auto inBounds = [&](const int& i, const int& j) {
            return i>=0 && i<R && j>=0 &&j<C;
        };
        while (!q.empty()) {
            Item curItem = q.front();
            q.pop();
            for (int dr = -1;dr<=1;dr++) {
                for (int dc =-1;dc<=1;dc++) {
                    if (abs(dc+dr)!=1) continue;
                    int nr = curItem.row+dr, nc = curItem.col+dc;
                    if (!inBounds(nr,nc) || grid[nr][nc]==-1) continue;
                    if (grid[nr][nc]>0) {
                        Item item;
                        item.init(nr,nc,grid[nr][nc],curItem.dist+1);
                        q.push(item);
                        if (inPrice(nr,nc)) {
                            heap.push(item);
                        }
                        if (heap.size()>k) {
                            heap.pop();
                        }
                    }
                    grid[nr][nc]=-1; // set as visited
                }
            }
        }
        vector<vector<int>> res;
        while (!heap.empty()) {
            Item item = heap.top();
            heap.pop();
            res.push_back({item.row,item.col});
        }
        reverse(res.begin(),res.end());
        return res;
    }
};
```

## 2147. Number of Ways to Divide a Long Corridor

### Solution: greedy solution with combinatorics 

```c++
const int MOD = 1e9+7;
const char SEAT = 'S';
class Solution {
public:
    int numberOfWays(string corridor) {
        int cntSeats = count_if(corridor.begin(),corridor.end(),[&](const auto& a) {return a==SEAT;}), n = corridor.size();
        if (cntSeats%2!=0 || cntSeats==0) {return 0;}
        long long way = 0, totalWays = 1;
        for (int i = 0, seats = 0;i<n;i++) {
            seats += (corridor[i]==SEAT);
            if (seats==3) {
                totalWays = (totalWays*way)%MOD;
                seats = 1;
                way = 0;
            }
            way += (seats==2);
        }
        return totalWays;
    }
};
```