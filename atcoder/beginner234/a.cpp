#include <bits/stdc++.h>
using namespace std;
int f(int x) {
    return x*x+2*x+3;
}
int main() {
    int t;
    cin>>t;
    int val = f(f(f(t)+t)+f(f(t)));
    cout<<val<<endl;
}