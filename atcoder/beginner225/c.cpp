#include <bits/stdc++.h>
using namespace std;
/*

*/
int main() {
    // freopen("inputs/input.txt","r",stdin);
    // freopen("outputs/output.txt","w",stdout);
    int N,M,b;
    cin>>N>>M;
    vector<vector<int>> B(N, vector<int>(M,0));
    for (int i = 0;i<N;i++) {
        for (int j=0;j<M;j++) {
            cin>>b;
            B[i][j]=b%7;
        }
    }
    bool found = true;
    auto check = [&](const int i, const int j) {
        return i>=0 && i<N && j>=0 && j<M;
    };
    for (int i = 0;i<N;i++) {
        for (int j = 0;j<M;j++) {
            int val = B[i][j];
            if (check(i,j+1)) {
                if ((val>0 && val<6 && B[i][j+1]!=val+1) || (val==6 && B[i][j+1]!=0) || val==0) {
                    found = false;
                    break;
                }
            }
            if (check(i+1,j)) {
                if (B[i+1][j]!=val) {
                    found = false;
                    break;
                }
            }
        }
    }
   
    if (found) {
        cout<<"Yes"<<endl;
    } else {
        cout<<"No"<<endl;
    }
}

/*
2 4
13 14 15 16
20 21 22 23
*/