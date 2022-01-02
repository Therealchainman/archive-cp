



D - Neighbors

Solution: Union Find

```c++
struct UnionFind {
    vector<int> parents, size;
    void init(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int ufind(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=ufind(parents[i]);
    }

    bool uunion(int i, int j) {
        i = ufind(i), j = ufind(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            return false;
        }
        return true;
    }
};

int main() {
    int N, M;
    cin >> N >> M;
    UnionFind ds;
    ds.init(N);
    unordered_map<int,int> counts;
    for (int i=0; i<M; i++) {
        int a, b;
        cin >> a >> b;
        if (ds.uunion(a-1,b-1)) {
            cout << "No" << endl;
            return 0;
        }
        counts[a]++;
        counts[b]++;
    }
    for (auto &[k,cnt] : counts) {
        if (cnt>2) {
            cout << "No" << endl;
            return 0;
        }
    }
    cout<<"Yes"<<endl;
    return 0;
}
```