#include <bits/stdc++.h>
using namespace std;
const int size = 25*6;
const int INF = 1e9;
int main() {
    freopen("inputs/input.txt", "r", stdin);
    vector<vector<int>> layers;
    string input;
    cin>>input;
    for (int i = 0; i < input.size(); i+=size) {
        vector<int> layer;
        for (int j = 0; j < size; j++) {
            layer.push_back(input[i+j]-'0');
        }
        layers.push_back(layer);
    }
    int mn = INF, mnidx = -1;
    for (int i = 0;i<layers.size();i++) {
        int cnt = accumulate(layers[i].begin(), layers[i].end(), 0, [](int& a, int& b) {
            return a + (b == 0);
        });
        if (cnt<mn) {
            mn = cnt;
            mnidx = i;
        }
    }
    int cntOnes = accumulate(layers[mnidx].begin(), layers[mnidx].end(), 0, [](int& a, int& b) {
        return a + (b == 1);
    }), cntTwos = accumulate(layers[mnidx].begin(), layers[mnidx].end(), 0, [](int& a, int& b) {
        return a + (b == 2);
    });
    cout<<cntOnes*cntTwos<<endl;
}