#include <bits/stdc++.h>
using namespace std;


int main() {
    freopen("inputs/input.txt", "r", stdin);
    string adj;
    unordered_map<string, vector<string>> orbits;
    queue<string> q;
    string target;
    unordered_set<string> visited;
    while (getline(cin, adj)) {
        int pos = adj.find(')');
        string u = adj.substr(0, pos), v = adj.substr(pos + 1);
        if (u=="YOU") {
            q.push(v);
            visited.insert(v);
            continue;
        }
        if (v=="YOU") {
            q.push(u);
            visited.insert(u);
            continue;
        }
        if (u=="SAN") {
            swap(v,target);
            continue;
        }
        if (v=="SAN") {
            swap(u,target);
            continue;
        }
        orbits[u].push_back(v);
        orbits[v].push_back(u);

    }
    int dist = 0;
    while (!q.empty()) {
        int sz = q.size();
        while (sz--) {
            string object = q.front();
            q.pop();
            if (object==target) {
                cout << dist << endl;
                break;
            }
            for (string nei : orbits[object]) {
                if (visited.find(nei)==visited.end()) {
                    q.push(nei);
                    visited.insert(nei);
                }
            }
        }
        dist++;
    }
}