# Croatian Open Competition in Informatics

# COCI 2024./2025. - Round #2

##

### Solution 1: 

```cpp

// namespace fs = std::filesystem;

// string name = "in.txt";

// fs::path create_path(const std::string& directory, const std::string& file_name) {
//     return fs::path(directory) / file_name;
// }

int N;
const vector<string> players = {"SONJA", "VIKTOR", "IGOR", "LEA", "MARINO"};
set<string> playedCards;
string bannedColors[5];
vector<pair<int, string>> paradoxes;

bool isParadox(int i, const string& card) {
    return bannedColors[i].find(card[0]) != string::npos || playedCards.count(card);
}

int cardNumber(const string& card) {
    return card[1] - '0';
}

void solve() {
    cin >> N;
    int pos = 0;
    for (int i = 0; i < N; i++) {
        char roundColor = 's';
        bool redSeen = false;
        int playerWinnerIdx = 0, value = 0;
        for (int j = 0; j < 5; j++) {
            string card;
            cin >> card;
            int playerIdx = (j + pos) % 5;
            if (isParadox(playerIdx, card)) {
                paradoxes.emplace_back(i + 1, players[playerIdx]);
            } else {
                if (roundColor == 's') {
                    roundColor = card[0];
                }
                if (card[0] != roundColor) {
                    bannedColors[playerIdx] += roundColor;
                }
                if (card[0] == 'C') {
                    if (!redSeen || (cardNumber(card) > value)) {
                        playerWinnerIdx = playerIdx;
                        value = cardNumber(card);
                    }
                    redSeen = true;
                } else if (card[0] == roundColor && !redSeen && cardNumber(card) > value) {
                    playerWinnerIdx = playerIdx;
                    value = cardNumber(card);
                }
                playedCards.insert(card);
            }
        }
        pos = playerWinnerIdx;
    }
    cout << paradoxes.size() << endl;
    for (const auto& [round, player] : paradoxes) {
        cout << round << " " << player << endl;
    }
}


signed main() {
    // fs::path input_path = create_path("inputs", name);
    // fs::path output_path = create_path("outputs", name);
    // ifstream input_file(input_path);
    // if (!input_file.is_open()) {
    //     std::cerr << "Error: Failed to open input file: " << input_path << endl;
    //     return 1;  // Exit with error if file cannot be opened
    // }
    // ofstream output_file(output_path);
    // if (!output_file.is_open()) {
    //     std::cerr << "Error: Failed to open output file: " << output_path << endl;
    //     return 1;  // Exit with error if file cannot be opened
    // }
    // cin.rdbuf(input_file.rdbuf());
    // cout.rdbuf(output_file.rdbuf());
    solve();
    cout.flush();
    return 0;
}

```

##

### Solution 1: 

```cpp
// namespace fs = std::filesystem;

// string name = "in.txt";

// fs::path create_path(const std::string& directory, const std::string& file_name) {
//     return fs::path(directory) / file_name;
// }

int N, D;
vector<int> values;
vector<int> dp;

void solve() {
    cin >> N >> D;
    values.assign(D + 1, 0);
    dp.assign(D + 1, 0);
    for (int i = 0; i < N; i++) {
        int p, t, o;
        cin >> p >> t >> o;
        for (int j = 1; p + j * t <= D; j++) {
            values[p + j * t] = max(values[p + j * t], j * o);
        }
    }
    for (int cap = 0; cap <= D; cap++) {
        for (int i = 1; i <= cap; i++) {
            dp[cap] = max(dp[cap], dp[cap - i] + values[i]);
        }
    }
    int ans = dp[D];
    cout << ans << endl;
}


signed main() {
    // fs::path input_path = create_path("inputs", name);
    // fs::path output_path = create_path("outputs", name);
    // ifstream input_file(input_path);
    // if (!input_file.is_open()) {
    //     std::cerr << "Error: Failed to open input file: " << input_path << endl;
    //     return 1;  // Exit with error if file cannot be opened
    // }
    // ofstream output_file(output_path);
    // if (!output_file.is_open()) {
    //     std::cerr << "Error: Failed to open output file: " << output_path << endl;
    //     return 1;  // Exit with error if file cannot be opened
    // }
    // cin.rdbuf(input_file.rdbuf());
    // cout.rdbuf(output_file.rdbuf());
    solve();
    cout.flush();
    return 0;
}
```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```


# COCI 2024./2025. - Round #3

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```
