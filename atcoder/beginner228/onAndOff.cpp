#include <bits/stdc++.h>
using namespace std;
/*

*/
bool isBetween(int a, int b, int c) {
    return a<=c && c<b; 
}
int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
    int S,T,X;
    cin>>S>>T>>X;
    if (S<T) {
        if (isBetween(S,T,X)) {
            cout<<"Yes"<<endl;
        } else {
            cout<<"No"<<endl;
        }
    } else {
        swap(S,T);
        if (isBetween(S,T,X)) {
            cout<<"No"<<endl;
        } else {
            cout<<"Yes"<<endl;
        }
    }

}