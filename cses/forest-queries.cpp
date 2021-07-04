#include <bits/stdc++.h>
using namespace std;

vector<vector<int>> prefixSum2D(vector<vector<int>>& matrix) {
    int R = matrix.size(), C = matrix[0].size();
    vector<vector<int>> prefix(R+1, vector<int>(C+1,0));
    for (int i = 1;i<=R;i++) {
        for (int j = 1;j<=C;j++) {
            prefix[i][j]=prefix[i-1][j]+prefix[i][j-1]-prefix[i-1][j-1]+matrix[i-1][j-1];
        }
    }
    return prefix;
}

int main() {
    int n, q;
    string line;
    cin>>n>>q;
    vector<vector<int>> matrix(n, vector<int>(n,0));
    for (int i = 0;i<n;i++) {
        cin>>line;
        for (int j = 0;j<n;j++) {
            matrix[i][j] = line[j]=='*' ? 1 : 0;
        }
    }
    vector<vector<int>> prefix = prefixSum2D(matrix);
    int x1,x2,y1,y2;
    while (q--) {
        cin>>y1>>x1>>y2>>x2;
        int minx = min(x1,x2), maxx = max(x1,x2), miny = min(y1,y2), maxy = max(y1,y2);
        int numForests = prefix[maxy][maxx]-prefix[miny-1][maxx]-prefix[maxy][minx-1]+prefix[miny-1][minx-1];
        cout<<numForests<<endl;
    }
}