


Example of regex match that is useful for some inputs

```cpp
regex plantRegex(R"(Plant\s+(\d+)\s+with thickness\s+(-?\d+):)");
regex branchRegex(R"(^-\s*branch to Plant\s+(\d+)\s+with thickness\s+(-?\d+))");
regex freeBranchRegex(R"(^-\s*free branch with thickness\s+(-?\d+))");

void solve() {
    vector<tuple<int, int, int>> edges;
    int N = 0;
    vector<vector<int>> queries;
    string line;
    int curPlant = -1;
    while (getline(cin, line)) {
        smatch match;
        if (regex_match(line, match, plantRegex)) {
            int plantId = stoi(match[1]);
            int thickness = stoi(match[2]);
            curPlant = plantId;
            N = max(N, plantId);
        } else if (regex_match(line, match, branchRegex)) {
            int fromPlantId = stoi(match[1]);
            int thickness = stoi(match[2]);
            edges.emplace_back(fromPlantId, curPlant, thickness);
        } else if (regex_match(line, match, freeBranchRegex)) {
            int thickness = stoi(match[1]);
            edges.emplace_back(0, curPlant, thickness);
        }
    }
```

Great way to read a line of integers into a vector

1 2 3 4 5 -> {1,2,3,4,5}

```cpp
istringstream iss(line);
vector<int> row;
int x;
while (iss >> x) {
    row.emplace_back(x);
}
```