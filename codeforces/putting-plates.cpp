#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, w,h;
    cin>>T;
    while (T--) {
        cin>>h>>w;
        vector<vector<int>> A(h,vector<int>(w,0)), B(h,vector<int>(w,0));
        int a = 0, b = 0;
        A[0][0]=1;
        if (w>1) {
            B[0][1]=1;
        } else {
            B[1][0]=1;
        }
        auto check = [&w,&h](const int i, const int j, const vector<vector<int>>& A) {
            if (i>0 && A[i-1][j]==1) {
                return false;
            }
            if (i<h-1 && A[i+1][j]==1) {
                return false;
            }
            if (j>0 && A[i][j-1]==1) {
                return false;
            }
            if (j<w-1 && A[i][j+1]==1) {
                return false;
            }
            if (i>0 && j>0 && A[i-1][j-1]==1) {
                return false;
            }
            if (i>0 && j<w-1 && A[i-1][j+1]==1) {
                return false;
            }
            if (i<h-1 && j>0 && A[i+1][j-1]==1) {
                return false;
            }
            if (i<h-1 && j<w-1 && A[i+1][j+1]==1) {
                return false;
            }
            return true;
        };
        // first row
        for (int i=0;i<w;i++) {
            if (check(0,i,A)) {
                A[0][i]=1;
                a++;
            }
            if (check(0,i,B)) {
                B[0][i]=1;
                b++;
            }
        }
        // last column
        for (int i = 0;i<h;i++) {
            if (check(i,w-1,A)) {
                A[i][w-1] = 1;
                a++;
            }
            if (check(i,w-1,B)) {
                B[i][w-1]=1;
                b++;
            }
        }
        // last row
        for (int i = w-1;i>=0;i--) {
            if (check(h-1,i,A)) {
                A[h-1][i]=1;
                a++;
            }
            if (check(h-1,i,B)) {
                B[h-1][i]=1;
                b++;
            }
        }
        // first column
        for (int i = h-1;i>=0;i--) {
            if (check(i,0,A)) {
                A[i][0]=1;
                a++;
            }
            if (check(i,0,B)) {
                B[i][0]=1;
                b++;
            }
        }
        if (b>a) {
            swap(A,B);
        }
        for (int i = 0;i<h;i++) {
            for (int j = 0;j<w;j++) {
                cout<<A[i][j];
            }
            cout<<endl;
        }

    }
}