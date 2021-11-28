#include <bits/stdc++.h>
using namespace std;
/*
binary search with set is my first solution

also this is apparently related to hashing with respect to linear probing in open addressing. 
linear probing is a method to deal with hash collision.  
When a hash function maps a key into a cell that is already occupied by a different key.  We use the linear probing
strategy by placing the new key into the closest following empty cell. 
*/
const int N = 1<<20;
long long A[N];

int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
    memset(A,-1,sizeof(A));
    long long Q,t,x;
    cin >> Q;
    vector<pair<long long, long long>> P(Q);
    for (int i = 0; i < Q; i++) {
        cin >> t >> x;
        P[i] = {t,x};
    }
    set<int> free; // indices that are free
    for (int i = 0;i<N;i++) {
        free.insert(i);
    }
    for (auto p: P) {
        t = p.first, x = p.second;
        if (t==1) {
            int h = x%N;
            auto it = free.lower_bound(h);
            if (it!=free.end()) {
                A[*it] = x;
                free.erase(it);
            } else {
                A[*free.begin()] = x;
                free.erase(free.begin());
            }
        } else {
            cout<<A[x%N]<<endl;
        }
    }
}