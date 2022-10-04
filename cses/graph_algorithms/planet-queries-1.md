# Planet Queries 1

## Solution 1:  binary jumping or binary lifting + sparse table + kth ancestor

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 2e5+5;
const int maxk = 32;

int sparse[maxn][maxk];

int kthAncestor(int x, int k) {
    for (int i = 0;i<maxk;i++) {
        if (k&(1<<i)) {
            x = sparse[x][i];
        }
    }
    return x+1;
}

int main() {
    ios_base::sync_with_stdio(false);
	cin.tie(NULL);
    int n, q, x,k;
    // freopen("input.txt","r",stdin);
    cin>>n>>q;
    vector<int> parents(n);
    for (int i = 0;i<n;i++) {
        cin>>parents[i];
        parents[i]--;
    }
    for (int j = 0;j<maxk;j++) {
        for (int i = 0;i<n;i++) {
            if (j==0) {
                sparse[i][j] = parents[i];
            } else if(sparse[i][j-1]!=-1) {
                int prev_ancestor = sparse[i][j-1];
                sparse[i][j] = sparse[prev_ancestor][j-1];
            }
        }
    }
    for (int i = 0;i<q;i++) {
        cin>>x>>k;
        x--;
        cout<<kthAncestor(x,k)<<endl;
    }
}
```