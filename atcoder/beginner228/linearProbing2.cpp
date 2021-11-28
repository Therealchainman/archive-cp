#include <bits/stdc++.h>
using namespace std;
/*

*/
constexpr int N = 1<<20;
constexpr int MASK = N-1;


int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
    int Q;
    cin >> Q;
    map<int,int> interval;
    map<int, long long> map;
    interval[N] = 0;
    while (Q--) {
        int t;
        long long x;
        cin>>t>>x;
        if (t==1) {
            int i = x&MASK;
        }
    }
}