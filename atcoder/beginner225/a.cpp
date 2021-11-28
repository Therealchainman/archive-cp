#include <bits/stdc++.h>
using namespace std;
/*

*/
int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
    string S;
    cin>>S;
    unordered_set<string> seen;
    sort(S.begin(),S.end());
    do {
        seen.insert(S);
    } while (next_permutation(S.begin(),S.end()));
    cout<<(int)seen.size()<<endl;
}
