#include <bits/stdc++.h>
using namespace std;
/*
Using the IntComputer and use 5 of them chained together where they provide an input to the intcomputer.  

Suppose you have an int computer A

We pass two inputs into the computer each time.  

The first number is the phase setting, and the second is the input from the previous computer.  For the first computer
you initialize with 0. 
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

struct IntComputer {
    vector<int> memory;
    void init() {
        string input;
        cin>>input;
        memory = getArray(input, ',');
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
    long long run(vector<long long>& inputs) {
        int i = 0, j = 0;
        long long output = 0;
        while (i<memory.size()) {
            string instructions = to_string(memory[i]);
            instructions = buildOpCode(instructions);
            int opcode = stoi(instructions.substr(3));
            if (opcode == 1) {
                int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
                int a = param1 ? memory[i+1] : memory[memory[i+1]], b = param2 ? memory[i+2] : memory[memory[i+2]];
                memory[memory[i+3]] = a + b;
                i += 4;
            } else if (opcode == 2) {
                int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
                int a = param1 ? memory[i+1] : memory[memory[i+1]], b = param2 ? memory[i+2] : memory[memory[i+2]];
                memory[memory[i+3]] = a * b;
                i += 4;
            } else if (opcode == 3) {
                memory[memory[i+1]] = inputs[j++];
                i += 2;
            } else if (opcode == 4) {
                int param = stoi(instructions.substr(2,1));
                output = param ? memory[i+1] : memory[memory[i+1]];
                return output;
                i += 2;
            } else if (opcode == 5) {
                int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
                int statement = param1 ? memory[i+1] != 0 : memory[memory[i+1]] != 0;
                i = statement ? (param2 ? memory[i+2] : memory[memory[i+2]]) : i+3;
            } else if (opcode == 6) {
                int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
                int statement = param1 ? memory[i+1] == 0 : memory[memory[i+1]] == 0;
                i = statement ? (param2 ? memory[i+2] : memory[memory[i+2]]) : i+3;
            } else if (opcode == 7) {
                int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
                int a = param1 ? memory[i+1] : memory[memory[i+1]], b = param2 ? memory[i+2] : memory[memory[i+2]];
                memory[memory[i+3]] = a < b ? 1 : 0;
                i += 4;
            } else if (opcode == 8) {
                int param1 = stoi(instructions.substr(2,1)), param2 = stoi(instructions.substr(1,1));
                int a = param1 ? memory[i+1] : memory[memory[i+1]], b = param2 ? memory[i+2] : memory[memory[i+2]];
                memory[memory[i+3]] = a == b ? 1 : 0;
                i += 4;
            } else if (opcode == 99) {
               break;
            }
        }
        return run(inputs)
    }
};
long long fullRun(IntComputer& comp, long long *phase, int n) {
    long long input = 0;
    for (int i = 0;i<n;i++) {
        long long p = *phase;
        vector<long long> inputs = {p, input};
        input = comp.run(inputs);
        phase++;
    }
    return input;
}
long long phases[5] = {5,6,7,8,9};
int main() {
    freopen("inputs/input.txt", "r", stdin);
    IntComputer computer;
    computer.init();
    long long maxOutput = 0;
    do {
        maxOutput = max(maxOutput, fullRun(computer, phases, 5));
    } while (next_permutation(phases, phases+5));
    cout<<maxOutput<<endl;
}