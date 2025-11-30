# Advent of Code 2019

## Day 1

### Part 1: 

```cpp
int main() {
    freopen("inputs/input.txt", "r", stdin);
    string tmp;
    int mass, sum = 0;
    while (getline(cin, tmp)) {
        mass = stoi(tmp);
        sum += (mass/3) -2;
    }
    cout<<sum<<endl;
}
```

### Part 2: 

```cpp
int main() {
    freopen("inputs/input.txt", "r", stdin);
    string tmp;
    int mass, sum = 0;
    while (getline(cin, tmp)) {
        mass = stoi(tmp);
        mass = (mass/3)-2;
        while (mass>0) {
            sum += mass;
            mass = (mass/3)-2;
        }
    }
    cout<<sum<<endl;
}
```

## Day 2

### Part 1: 

```cpp
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
```

### Part 2: 

```cpp
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
```

## Day 3

### Part 1: 

```cpp
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
```

### Part 2: 

```cpp
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

void getLocationsCrossed(vector<pair<string, int>>& wire, vector<vector<int>>& path) {
    #define direction first
    #define distance second
    path.push_back({0,0,0});
    for (auto &command : wire) {
        if (command.direction == "R") {
            for (int i = 0; i < command.distance; i++) {
                path.push_back({path.back()[0] + 1, path.back()[1],path.back()[2]+1});
            }
        } else if (command.direction == "L") {
            for (int i = 0; i < command.distance; i++) {
                path.push_back({path.back()[0] - 1, path.back()[1],path.back()[2]+1});
            }
        } else if (command.direction == "U") {
            for (int i = 0; i < command.distance; i++) {
                path.push_back({path.back()[0], path.back()[1]+1,path.back()[2]+1});
            }
        } else if (command.direction == "D") {
            for (int i = 0; i < command.distance; i++) {
                path.push_back({path.back()[0], path.back()[1]-1,path.back()[2]+1});
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
    vector<vector<int>> wire1Path, wire2Path;
    getLocationsCrossed(wire1, wire1Path);
    getLocationsCrossed(wire2, wire2Path);
    sort(wire1Path.begin(), wire1Path.end());
    sort(wire2Path.begin(), wire2Path.end());
    int minSteps = INF;
    for (int i = 0, j = 0;i<wire1Path.size() && j<wire2Path.size();) {
        auto vec1 = wire1Path[i], vec2 = wire2Path[j];
        pair<int,int> point1 = {vec1[0], vec1[1]}, point2 = {vec2[0], vec2[1]};
        int steps1 = vec1[2], steps2 = vec2[2];
        if (point1==CENTRAL_POINT) {
            i++;
            continue;
        } else if (point2==CENTRAL_POINT) {
            j++;
            continue;
        }
        if (point1==point2) {
            minSteps = min(minSteps, steps1+steps2);
            i++;
        } else if (point1 < point2) {
            i++;
        } else {
            j++;
        }
    }
    cout<<minSteps<<endl;
}
```

## Day 4

### Part 1: 

```cpp
int main() {
    freopen("inputs/input.txt", "r", stdin);
    string input;
    cin>>input;
    int pos = input.find("-");
    int start = stoi(input.substr(0, pos));
    int end = stoi(input.substr(pos+1));
    int count = 0;
    for (int i = start;i<=end;i++) {
        bool adj = false, nonDecreasing = true;
        string s = to_string(i);
        for (int j = 1;j<s.size();j++) {
            if (s[j]==s[j-1]) {
                adj = true;
            } 
            if (s[j]<s[j-1]) {
                nonDecreasing = false;
            }
        }
        count += adj && nonDecreasing;
    }
    cout<<count<<endl;
}
```

### Part 2: 

```cpp
int main() {
    freopen("inputs/input.txt", "r", stdin);
    string input;
    cin>>input;
    int pos = input.find("-");
    int start = stoi(input.substr(0, pos));
    int end = stoi(input.substr(pos+1));
    int count = 0;
    for (int i = start;i<=end;i++) {
        bool adj = false, nonDecreasing = true;
        string s = to_string(i);
        int prev = s[0]-'0', cnt = 1;
        for (int j = 1;j<s.size();j++) {
            cnt += (s[j]-'0' == prev);
            if (s[j]-'0' != prev || j==s.size()-1) {
                adj |= (cnt==2);
                cnt = 1;
                prev = s[j]-'0';
            }
            if (s[j]<s[j-1]) {
                nonDecreasing = false;
            }
        }
        count += adj && nonDecreasing;
    }
    cout<<count<<endl;
}
```

## Day 5

### Part 2: 

```cpp
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
```

## Day 6

### Part 1: 

```cpp
unordered_map<string, vector<string>> orbits;
int dfs(string object = "COM", int depth = 1) {
    int numOrbits = 0;
    for (string orb : orbits[object]) {
        numOrbits += dfs(orb, depth+1) + depth;
    }
    return numOrbits;
}
int main() {
    freopen("inputs/input.txt", "r", stdin);
    string adj;
    while (getline(cin, adj)) {
        int pos = adj.find(')');
        string u = adj.substr(0, pos), v = adj.substr(pos + 1);
        orbits[u].push_back(v);
    }
    int res = dfs();
    cout<<res<<endl;
}
```

### Part 2: 

```cpp
int main() {
    freopen("inputs/input.txt", "r", stdin);
    string adj;
    unordered_map<string, vector<string>> orbits;
    queue<string> q;
    string target;
    unordered_set<string> visited;
    while (getline(cin, adj)) {
        int pos = adj.find(')');
        string u = adj.substr(0, pos), v = adj.substr(pos + 1);
        if (u=="YOU") {
            q.push(v);
            visited.insert(v);
            continue;
        }
        if (v=="YOU") {
            q.push(u);
            visited.insert(u);
            continue;
        }
        if (u=="SAN") {
            swap(v,target);
            continue;
        }
        if (v=="SAN") {
            swap(u,target);
            continue;
        }
        orbits[u].push_back(v);
        orbits[v].push_back(u);

    }
    int dist = 0;
    while (!q.empty()) {
        int sz = q.size();
        while (sz--) {
            string object = q.front();
            q.pop();
            if (object==target) {
                cout << dist << endl;
                break;
            }
            for (string nei : orbits[object]) {
                if (visited.find(nei)==visited.end()) {
                    q.push(nei);
                    visited.insert(nei);
                }
            }
        }
        dist++;
    }
}
```

## Day 7

### Part 2: 

```cpp
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
```

## Day 8

### Part 2: 

```cpp
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
```
