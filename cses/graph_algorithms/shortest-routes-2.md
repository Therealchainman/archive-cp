# Shortest Routes I

## Solution: Floyd warshall algorithm to compute the shortest distance from all nodes to all nodes, ASSP => All Sources Shortest Paths

Compared to problem shortest routes 1, which is SSSP Single source shortest paths

```py
from math import inf
def main():
    num_cities, num_roads, num_queries = map(int,input().split())
    dist = [[inf]*(num_cities+1) for _ in range(num_cities+1)]
    for i in range(num_cities+1):
        dist[i][i] = 0
    for _ in range(num_roads):
        city1, city2, length = map(int,input().split())
        dist[city1][city2] = dist[city2][city1] = length
    for k in range(1, num_cities+1):
        for i in range(1,num_cities+1):
            if dist[i][k] == inf: continue
            for j in range(1,num_cities+1):
                if dist[k][j] == inf: continue
                dist[i][j] = min(dist[i][j], dist[i][k]+dist[j][k])
    return "\n".join(map(str, (dist[city1][city2] if dist[city1][city2] != inf else -1 \
    for city1, city2 in [map(int,input().split()) for _ in range(num_queries)])))

if __name__ == '__main__':
    print(main())
```

```c++
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
```