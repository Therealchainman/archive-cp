Geekland Junction

First solution uses queue and bfs algorithm and an unordered set. 

```c++
int junction(int N, vector<int> p){
    vector<vector<int>> graph(N);
    queue<int> q;
    for (int i = 0;i<N;i++) {
        if (p[i]!=-1) {
            graph[p[i]].push_back(i);
        } else {
            q.push(i); // root node
        }
    }
    unordered_set<int> lastCities;
    while (!q.empty()) {
        int sz = q.size();
        lastCities.clear();
        while (sz--) {
            int city = q.front();
            q.pop();
            lastCities.insert(city);
            for (int nei : graph[city]) {
                q.push(nei);
            }
        }
    }
    while (lastCities.size()>1) {
        int sz = lastCities.size();
        unordered_set<int> ncities;
        for (auto city : lastCities) {
            ncities.insert(p[city]);
        }
        lastCities = ncities;
    }
    return *lastCities.begin();
}
```