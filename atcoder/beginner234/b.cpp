#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, x, y;
    cin>>N;
    cout<<setprecision(20);
    vector<pair<int,int>> points;
    for (int i = 0;i<N;i++) {
        cin>>x>>y;
        points.emplace_back(x,y);
    }
    double best = 0.0;
    #define x first
    #define y second
    for (int i = 0;i<N;i++) {
        for (int j = i+1;j<N;j++) {
            best = max(best, sqrt((points[i].x-points[j].x)*(points[i].x-points[j].x)+(points[i].y-points[j].y)*(points[i].y-points[j].y)));
        }
    }
    cout<<best<<endl;
}