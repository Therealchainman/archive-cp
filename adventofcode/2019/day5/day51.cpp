#include <bits/stdc++.h>
using namespace std;
/*
Manually provide 1 as the input into the intcomputer, it only requests an input once.  
The output prints to screen and the last output is the diagnostic code.  If all the outputs prior were 
0 then the diagnostic program ran successfully. 
*/
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

string buildOpCode(string& old) {
    int sz = 5 - old.size();
    string s = "";
    while (sz--) {
        s += "0";
    }
    s += old;
    return s;
}

int main() {
    string input;
    cin>>input;
    vector<int> ints = getArray(input, ',');
    int i = 0;
    while (i<ints.size()) {
        string instructions = to_string(ints[i]);
        instructions = buildOpCode(instructions);
        int opcode = stoi(instructions.substr(3));
        if (opcode == 1) {
            int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
            int a = param1 ? ints[i+1] : ints[ints[i+1]], b = param2 ? ints[i+2] : ints[ints[i+2]];
            ints[ints[i+3]] = a + b;
            i += 4;
        } else if (opcode == 2) {
            int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
            int a = param1 ? ints[i+1] : ints[ints[i+1]], b = param2 ? ints[i+2] : ints[ints[i+2]];
            ints[ints[i+3]] = a * b;
            i += 4;
        } else if (opcode == 3) {
            int input;
            printf("Enter input: ");
            cin>>input;
            ints[ints[i+1]] = input;
            i += 2;
        } else if (opcode == 4) {
            int param = stoi(instructions.substr(2,1));
            printf("output: %d\n", param ? ints[i+1] : ints[ints[i+1]]);
            flush(cout);
            i += 2;
        } else if (opcode == 99) {
            break;
        }
    }
}