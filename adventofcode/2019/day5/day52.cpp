#include <bits/stdc++.h>
using namespace std;
/*
Adding more opcodes
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
        } else if (opcode == 5) {
            int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
            int statement = param1 ? ints[i+1] != 0 : ints[ints[i+1]] != 0;
            i = statement ? (param2 ? ints[i+2] : ints[ints[i+2]]) : i+3;
        } else if (opcode == 6) {
            int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
            int statement = param1 ? ints[i+1] == 0 : ints[ints[i+1]] == 0;
            i = statement ? (param2 ? ints[i+2] : ints[ints[i+2]]) : i+3;
        } else if (opcode == 7) {
            int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
            int a = param1 ? ints[i+1] : ints[ints[i+1]], b = param2 ? ints[i+2] : ints[ints[i+2]];
            ints[ints[i+3]] = a < b ? 1 : 0;
            i += 4;
        } else if (opcode == 8) {
            int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
            int a = param1 ? ints[i+1] : ints[ints[i+1]], b = param2 ? ints[i+2] : ints[ints[i+2]];
            ints[ints[i+3]] = a == b ? 1 : 0;
            i += 4;
        } else if (opcode == 99) {
            break;
        }
    }
}