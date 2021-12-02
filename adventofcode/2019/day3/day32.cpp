#include <bits/stdc++.h>
using namespace std;
/*
Now I want to find the fewest combined steps to the intersection. 
*/

const int INF = 1e9;
const pair<int,int> CENTRAL_POINT = {0,0};
vector<pair<string, int>> getArray(string &str, char delim) {
  vector<pair<string, int>> nodes;
  stringstream ss(str);
  string tmp;
  while (getline(ss, tmp, delim)) {
    if (tmp.empty()) {continue;}
    nodes.emplace_back(tmp.substr(0,1), stoi(tmp.substr(1)));
  }
  return nodes;
}
int manhattanDistance(const pair<int, int> &a, const pair<int, int> &b) {
  return abs(a.first - b.first) + abs(a.second - b.second);
}

void getLocationsCrossed(vector<pair<string, int>>& wire, vector<vector<int>>& path) {
    #define direction first
    #define distance second
    path.push_back({0,0,0});
    for (auto &command : wire) {
        if (command.direction == "R") {
            for (int i = 0; i < command.distance; i++) {
                path.push_back({path.back()[0] + 1, path.back()[1],path.back()[2]+1});
            }
        } else if (command.direction == "L") {
            for (int i = 0; i < command.distance; i++) {
                path.push_back({path.back()[0] - 1, path.back()[1],path.back()[2]+1});
            }
        } else if (command.direction == "U") {
            for (int i = 0; i < command.distance; i++) {
                path.push_back({path.back()[0], path.back()[1]+1,path.back()[2]+1});
            }
        } else if (command.direction == "D") {
            for (int i = 0; i < command.distance; i++) {
                path.push_back({path.back()[0], path.back()[1]-1,path.back()[2]+1});
            }
        }
    }
}
int main() {
    freopen("inputs/input.txt", "r", stdin);
    string line;
    cin>>line;
    vector<pair<string, int>> wire1 = getArray(line, ',');
    cin>>line;
    vector<pair<string, int>> wire2 = getArray(line, ',');
    vector<vector<int>> wire1Path, wire2Path;
    getLocationsCrossed(wire1, wire1Path);
    getLocationsCrossed(wire2, wire2Path);
    sort(wire1Path.begin(), wire1Path.end());
    sort(wire2Path.begin(), wire2Path.end());
    int minSteps = INF;
    for (int i = 0, j = 0;i<wire1Path.size() && j<wire2Path.size();) {
        auto vec1 = wire1Path[i], vec2 = wire2Path[j];
        pair<int,int> point1 = {vec1[0], vec1[1]}, point2 = {vec2[0], vec2[1]};
        int steps1 = vec1[2], steps2 = vec2[2];
        if (point1==CENTRAL_POINT) {
            i++;
            continue;
        } else if (point2==CENTRAL_POINT) {
            j++;
            continue;
        }
        if (point1==point2) {
            minSteps = min(minSteps, steps1+steps2);
            i++;
        } else if (point1 < point2) {
            i++;
        } else {
            j++;
        }
    }
    cout<<minSteps<<endl;
}