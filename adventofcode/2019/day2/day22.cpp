#include <bits/stdc++.h>
using namespace std;
/*
what produces this value? 19690720
*/
const int N = 19690720;
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
int getCode(vector<int>& intCodes) {
    int noun, verb;
    vector<int> originalIntCodes = intCodes;
    for (noun=0;noun<100;noun++) {
        for (verb=0;verb<100;verb++) {
            intCodes = originalIntCodes;
            intCodes[1] = noun;
            intCodes[2] = verb;
            for (int i = 0;i<intCodes.size();i+=4) {
                if (intCodes[i]==1) {
                    int a = intCodes[i+1], b = intCodes[i+2], c = intCodes[i+3];
                    intCodes[c] = intCodes[a] + intCodes[b];
                } else if (intCodes[i]==2) {
                    int a = intCodes[i+1], b = intCodes[i+2], c = intCodes[i+3];
                    intCodes[c] = intCodes[a] * intCodes[b];
                } else if (intCodes[i]==99) {
                    break;
                }
            }
            if (intCodes[0] == N) {
                return 100*noun+verb;
            }
        }
    }
    return -1;
}

int main() {
    freopen("inputs/input.txt", "r", stdin);
    freopen("outputs/output.txt", "w", stdout);
    string input;
    cin>>input;
    vector<int> intCodes = getArray(input, ',');
    cout<<getCode(intCodes)<<endl;
}