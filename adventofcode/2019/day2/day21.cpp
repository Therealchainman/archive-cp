#include <bits/stdc++.h>
using namespace std;

vector<int> getArray(string &str, char delim) {
  vector<int> nodes;
  stringstream ss(str);
  string tmp;
  while (getline(ss, tmp, delim)) {
    if (tmp.empty()) {continue;}
    nodes.push_back(stoi(tmp));
  }
  return nodes;
}

int main() {
    freopen("inputs/input.txt", "r", stdin);
    string input;
    cin>>input;
    vector<int> ints = getArray(input, ',');
    ints[1]=12;
    ints[2]=2;
    for (int i = 0;i<ints.size();i+=4) {
        if (ints[i]==1) {
            int a = ints[i+1], b = ints[i+2], c = ints[i+3];
            ints[c] = ints[a] + ints[b];
        } else if (ints[i]==2) {
            int a = ints[i+1], b = ints[i+2], c = ints[i+3];
            ints[c] = ints[a] * ints[b];
        } else if (ints[i]==99) {
            break;
        }
    }
    cout<<ints[0]<<endl;
}