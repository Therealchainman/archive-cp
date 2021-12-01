#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <sstream>
#include <iostream>

using namespace std;
typedef vector<int> vec;
typedef long long int lli;

lli countTrees(vector<string> map, int dx, int dy) {
    lli ans = 0;
    int C = map[0].size(), R = map.size();
    int c = dx;
    for (int r = dy;r<R;r+=dy) {
        if (map[r][c%C]=='#') {
            ans++;
        } 
        c+=dx;
    }
    return ans;
}