#include <bits/stdc++.h>
using namespace std;

int main() {
    freopen("inputs/input.txt", "r", stdin);
    string line;
    vector<vector<pair<int,int>>> lineSegments;
    while (getline(cin, line)) {
        int pos = line.find(" -> ");
        int pos1 = line.find(',');
        int pos2 = line.find(',', pos1 + 1);
        string startx = line.substr(0, pos1), starty = line.substr(pos1 + 1, pos - pos1- 1), endx = line.substr(pos + 4, pos2-(pos+3)-1), endy = line.substr(pos2+1);
        lineSegments.push_back({{stoi(startx), stoi(starty)}, {stoi(endx), stoi(endy)}});
    }
    map<pair<int,int>, int> freq;
    for (auto &lineSegment : lineSegments) {
        auto start = lineSegment[0], end = lineSegment[1];
        #define x first
        #define y second
        int deltaX = end.x > start.x ? 1 : end.x < start.x ? -1 : 0;
        int deltaY = end.y > start.y ? 1 : end.y < start.y ? -1 : 0;
        for (int ix = start.x, iy = start.y; ix != end.x || iy != end.y; ix += deltaX, iy += deltaY) {
            freq[{ix, iy}]++;
        }
        freq[{end.x, end.y}]++;
    }
    int cnt = 0;
    for (auto point: freq) {
        #define count second
        cnt += (point.count >= 2);
    }
    cout<<cnt<<endl;
}
