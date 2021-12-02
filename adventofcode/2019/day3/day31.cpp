#include <bits/stdc++.h>
using namespace std;
/*
Crossed Wires
The problem is asking to find the closest intersection to the central point or origin of the two wires via
the manhattan metric. 
Time complexity is about O(nlogn) for the length of the wires if I store every single location the wire crosses in a grid. 
Since it is only about 100k grid locations that each wire crosses I can store it in a vector, and I get two vectors of size 150k that 
store every single (x,y) location the wire crosses.  So then I sort these two vectors so that I can use two pointer technique to
iterate over it in O(n). 
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

void getLocationsCrossed(vector<pair<string, int>>& wire, vector<pair<int, int>>& path) {
    #define direction first
    #define distance second
    #define x first
    #define y second
    path.push_back(CENTRAL_POINT);
    for (auto &command : wire) {
        if (command.direction == "R") {
            for (int i = 0; i < command.distance; i++) {
                path.emplace_back(path.back().x + 1, path.back().y);
            }
        } else if (command.direction == "L") {
            for (int i = 0; i < command.distance; i++) {
                path.emplace_back(path.back().x - 1, path.back().y);
            }
        } else if (command.direction == "U") {
            for (int i = 0; i < command.distance; i++) {
                path.emplace_back(path.back().x, path.back().y + 1);
            }
        } else if (command.direction == "D") {
            for (int i = 0; i < command.distance; i++) {
                path.emplace_back(path.back().x, path.back().y - 1);
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
    vector<pair<int, int>> wire1Path, wire2Path;
    getLocationsCrossed(wire1, wire1Path);
    getLocationsCrossed(wire2, wire2Path);
    sort(wire1Path.begin(), wire1Path.end());
    sort(wire2Path.begin(), wire2Path.end());
    int minDist = INF;
    for (int i = 0, j = 0;i<wire1Path.size() && j<wire2Path.size();) {
        pair<int,int> point1 = wire1Path[i], point2 = wire2Path[j];
        if (point1==CENTRAL_POINT) {
            i++;
            continue;
        } else if (point2==CENTRAL_POINT) {
            j++;
            continue;
        }
        if (point1==point2) {
            minDist = min(minDist, manhattanDistance(CENTRAL_POINT, point1));
            i++;
        } else if (point1 < point2) {
            i++;
        } else {
            j++;
        }
    }
    cout<<minDist<<endl;
}