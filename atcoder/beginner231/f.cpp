#include <bits/stdc++.h>
using namespace std;
/*
I'd have to write out the solution in brute force which is 
O(n^2) to find a pattern to optimize it.  Cause I don't see anything yet.  
*/

int main() {
    int N;
    vector<pair<int,int>> A(N), B(N);
    #define value first
    #define index second
    for (int i = 0;i<N;i++) {
        int a, b;
        cin>>a>>b;
        A[i].first = a;
        A[i].second = i;
        B[i].first = b;
        B[i].second = i;
    }
    sort(A.rbegin(), A.rend());
    sort(B.rbegin(), B.rend());
    vector<int> where(N);
    for (int i = 0;i<N;i++) {
        where[B[i].index] = i;
    }
    set<int> removed;
    int ans = 0;
    for (int i = 0;i<N;i++) {
        int cand = A[i].index; // this is the cand, where is it located in B I can get below
        int j = where[cand]; // index in B so I want all to the right of it
        int num = N-j;
        removed.insert(j); // marked as removed
    }
    cout<<ans<<endl;
}