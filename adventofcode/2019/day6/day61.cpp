#include <bits/stdc++.h>
using namespace std;
unordered_map<string, vector<string>> orbits;
int dfs(string object = "COM", int depth = 1) {
    int numOrbits = 0;
    for (string orb : orbits[object]) {
        numOrbits += dfs(orb, depth+1) + depth;
    }
    return numOrbits;
}
int main() {
    freopen("inputs/input.txt", "r", stdin);
    string adj;
    while (getline(cin, adj)) {
        int pos = adj.find(')');
        string u = adj.substr(0, pos), v = adj.substr(pos + 1);
        orbits[u].push_back(v);
    }
    int res = dfs();
    cout<<res<<endl;
}