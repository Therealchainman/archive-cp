#include <bits/stdc++.h>
using namespace std;
/*
part 1
*/
const int INF = 1e8;
int main() {
    freopen("inputs/input1.txt", "r", stdin);
    freopen("outputs/output1.txt", "w", stdout);
    string tmp;
    int cnt = 0, depth, pDepth = INF;
    while(getline(cin, tmp)) {
        depth = stoi(tmp);
        cnt += (depth>pDepth);
        pDepth = depth;
    }
    cout<<cnt<<endl;
}

