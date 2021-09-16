#include <bits/stdc++.h>
using namespace std;


int minimizeTheDifference(vector<vector<int>>& mat, int target) {
    unordered_set<int> seen;
    seen.insert(0);
    int R = mat.size(), C = mat[0].size();
    for (int i = 0;i<R;i++) {
        unordered_set<int> next;
        int goal = 1e4;
        for (int j = 0;j<C;j++) {
            for (auto &v : seen) {
                int cand = v+mat[i][j];
                if (cand<target) {
                    next.insert(cand);
                } else if (cand<goal) {
                    next.erase(goal);
                    next.insert(cand);
                    goal = cand;
                }
            }
        }
        seen=next;
    }
    int ans = 1e4;
    for (auto &v : seen) {
        ans = min(ans, abs(target-v));
    }
    return ans;
}

int main() {
    freopen("in.txt", "r", stdin);
    freopen("out.txt", "w", stdout);

    cout<<minimizeTheDifference(mat, target)<<endl;
}

/*
The fastest solution is to use bitsets in c++, 
It is good idea to be familiar with bitsets because it is a space
efficient method to store an array of bits when you know the size
if the size needs to be dynamic you can use vector<bool>
*/

int minimizeTheDifference(vector<vector<int>>& mat, int target) {
    int R = mat.size(), C = mat[0].size();
    bitset<4901> bs;
    bs[0]=1;
    for (int i = 0;i<R;i++) {
        bitset<4901> bs2;
        for (int j = 0;j<C;j++) {
            bs2 |= (bs<<mat[i][j]);
        }
        swap(bs,bs2);
    }
    for (int i = target, j=target;i>=0 || j<=4900;i--,j++) {
        if (i>=0 && bs[i]) {
            return target-i;
        } 
        if (j<=4900 && bs[j]) {
            return j-target;
        }
    }
    return -1;
}