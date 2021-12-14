#include <bits/stdc++.h>
using namespace std;
/*
part 2
*/
int main() {
    freopen("inputs/input1.txt", "r", stdin);
    freopen("outputs/output2.txt", "w", stdout);
    string tmp;
    int cnt = 0, depth;
    vector<int> depths;
    while(getline(cin, tmp)) {
        depth = stoi(tmp); 
        depths.push_back(depth);
    }
    for(int i=3; i<depths.size(); i++) {
        cnt += (depths[i]>depths[i-3]);
    }
    cout<<cnt<<endl;
}

