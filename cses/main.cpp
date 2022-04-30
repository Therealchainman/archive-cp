#include <bits/stdc++.h>
using namespace std;
const long long INF = 1e16;
int main() {
    int num_cities, num_roads, num_queries, city1, city2;
    long long road_length;
    cin>>num_cities>>num_roads>>num_queries;
    vector<vector<long long>> dist(num_cities+1, vector<long long>(num_cities+1, INF));
    for (int i = 1;i<=num_cities;i++) {
        dist[i][i] = 0;
    }
    for (int i = 0;i<num_roads;i++) {
        cin>>city1>>city2>>road_length;
        dist[city1][city2] = min(dist[city1][city2], road_length);
        dist[city2][city1] = min(dist[city2][city1], road_length);
    }
    for (int k = 1;k<=num_cities;k++) {
        for (int i = 1;i<=num_cities;i++) {
            if (dist[i][k] == INF) continue;
            for (int j = 1;j<=num_cities;j++) {
                if (dist[k][j] == INF) continue;
                
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[j][k]);
            }
        }
    }
    for (int i = 0;i<num_queries;i++) {
        cin>>city1>>city2;
        cout<<(dist[city1][city2] < INF ? dist[city1][city2]: -1)<<endl;
    }
}