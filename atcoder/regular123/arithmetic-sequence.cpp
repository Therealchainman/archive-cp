#include <bits/stdc++.h>
using namespace std;
int main() {
    long long A1,A2,A3;
    cin>>A1>>A2>>A3;
    long long D = A2-A1, R = A3-A2;
    long long diff = abs(D-R), opers;
    if (D>R) {
        opers = diff;
    } else {
        opers = diff/2;
        
        if (diff%2!=0) {
            opers+=2;
        }
    }
    cout<<opers<<endl;
}