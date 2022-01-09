# Leetcode Weekly Contest 275

## 2133. Check if Every Row and Column Contains All Numbers

### Solution: rows and columns count O(n^2) and O(n) memory

```c++
bool checkValid(vector<vector<int>>& matrix) {
    int n = matrix.size();
    int rows[n+1], cols[n+1];
    for (int i = 0;i<n;i++) {
        memset(rows,0,sizeof(rows));
        memset(cols,0,sizeof(cols));
        for (int j = 0;j<n;j++) {
            if (++rows[matrix[j][i]]>1) {return false;}
            if (++cols[matrix[i][j]]>1) {return false;}
        }
    }
    return true;
}
```



## 5979. Earliest Possible Day of Full Bloom

### Solution: sorting - greedy plant the flower seeds that have the longest growing time first. 


```c++
int earliestFullBloom(vector<int>& P, vector<int>& G) {
    int n = P.size();
    vector<int> plants(n);
    iota(plants.begin(), plants.end(),0);
    sort(plants.begin(),plants.end(),[&](const auto& i, const auto& j) {
        if (G[i]!=G[j]) {
            return G[i]>G[j];
        } 
        return P[i]>P[j];
    });
    int finish = 0;
    for (int i = 0, time = 0;i<n;i++) {
        int index = plants[i];
        time += P[index];
        finish = max(finish, time+G[index]);
    }
    return finish;
}
```

### Solution: multiset with custom comparator for plant sort descending order for growing and if tied, ascending order for planting.


```c++
struct Plant {
    int grow,seed;
    void init(int g, int s) {
        grow=g;
        seed=s;
    }
    bool operator<(const Plant& b) const {
        if (grow!=b.grow) {
            return grow>b.grow;
        }
        return seed<b.seed;
    }
};
class Solution {
public:
    int earliestFullBloom(vector<int>& P, vector<int>& G) {
        int n = P.size();
        multiset<Plant> plantSet;
        for (int i = 0;i<n;i++) {
            Plant pl;
            pl.init(G[i],P[i]);
            plantSet.insert(pl);
        }
        int finish = 0, time = 0;
        for (auto& plant : plantSet) {
            time += plant.seed;
            finish = max(finish, time+plant.grow);
        }
        return finish;
    }
};
```