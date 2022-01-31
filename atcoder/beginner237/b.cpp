#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, a;
    cin >> H >> W;
    vector<vector<int>> A(H, vector<int>(W,0)), B(W, vector<int>(H,0));
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            cin >> a;
            A[i][j] = a;
        }
    }

    for (int i = 0;i<W;i++) {
        for (int j = 0;j<H;j++) {
            B[i][j] = A[j][i];
        }
    }
    for (int i = 0;i<W;i++) {
        for (int j = 0;j<H;j++) {
            cout << B[i][j] << " ";
        }
        cout<<endl;
    }
}