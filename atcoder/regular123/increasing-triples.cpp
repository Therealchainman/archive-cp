#include <bits/stdc++.h>
using namespace std;
void fill(int n, vector<int>& arr) {
    int tmp;
    for (int i = 0;i<n;i++) {
        cin>>tmp;
        arr[i]=tmp;
    }
}
void fillmultiset(multiset<int>& ms, vector<int>& arr) {
    for (int x : arr) {
        ms.insert(x);
    }
}
int main() {
    int N,tmp;
    cin>>N;
    vector<int> A(N,0),B(N,0),C(N,0);
    fill(N,A);
    fill(N,B);
    fill(N,C);
    sort(A.begin(),A.end());
    sort(B.begin(),B.end());
    sort(C.begin(),C.end());
    multiset<int> mb,mc;
    fillmultiset(mb,B);
    fillmultiset(mc,C);
    vector<int> arr;
    for (int i=0;i<N;i++) {
        auto x = mb.upper_bound(A[i]);
        if (x==mb.end()) {
            break;
        }
        arr.push_back(*x);
        mb.erase(x);
    }
    int cnt = 0;
    for (int i = 0;i<arr.size();i++) {
        auto x = mc.upper_bound(arr[i]);
        if (x==mc.end()) {
            break;
        }
        cnt++;
        mc.erase(x);
    }
    cout<<cnt<<endl;
}