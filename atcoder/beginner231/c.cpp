#include <bits/stdc++.h>
using namespace std;


int main() {
    int N, Q;
    cin >> N >> Q;
    vector<int> A(N);
    for (int i = 0; i < N; i++) {
        int a;
        cin >> a;
        A[i] = a;
    }
    sort(A.begin(), A.end());
    while (Q--) {
        int x;
        cin>>x;
        int i = lower_bound(A.begin(), A.end(), x) - A.begin();
        cout<< N-i<<endl;
    }
}