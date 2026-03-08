# Everbody Codes 2026

# Melody Made of Code [ No. 3 ]

# Quest 1: Scales, Bags and a Bit of a Mess

## Part 1

### Solution 1: regex, string manipulation, greedy

```cpp
regex pattern(R"((\d+):([rR]+)\s+([gG]+)\s+([bB]+))");

bool isGreater(const string& s, const string& t) {
    int N = s.size();
    assert(s.size() == t.size());
    for (int i = 0; i < N; ++i) {
        if (isupper(s[i]) && islower(t[i])) return true;
        if (islower(s[i]) && isupper(t[i])) return false;
    }
    return false;
}

void solve() {
    string line;
    int ans = 0;
    while (getline(cin, line)) {
        smatch match;
        if (regex_match(line, match, pattern)) {
            int id = stoi(match[1]);
            string red = match[2], green = match[3], blue = match[4];
            if (isGreater(green, red) && isGreater(green, blue)) ans += id;
        }
    }
    cout << ans << endl;
}

```

## Part 2

### Solution 1: evaluating binary values, sorting, reverse

```cpp
regex pattern(R"((\d+):([rR]+)\s+([gG]+)\s+([bB]+)\s+([sS]+))");

struct Scale {
    int id, colorVal, shineVal;
    Scale(int id, int colorVal, int shineVal) : id(id), colorVal(colorVal), shineVal(shineVal) {}
    bool operator<(const Scale& other) const {
        if (shineVal != other.shineVal) return shineVal > other.shineVal;
        return colorVal < other.colorVal;
    }
};

int eval(string s) {
    int res = 0, N = s.size();
    reverse(s.begin(), s.end());
    for (int i = 0; i < N; ++i) {
        if (isupper(s[i])) res += (1 << i);
    }
    return res;
}

void solve() {
    string line;
    int ans = 0, best = 0;
    vector<Scale> scales;
    while (getline(cin, line)) {
        smatch match;
        if (regex_match(line, match, pattern)) {
            int id = stoi(match[1]);
            string red = match[2], green = match[3], blue = match[4], shine = match[5];
            int shineVal = eval(shine);
            int val = eval(red) + eval(green) + eval(blue);
            scales.emplace_back(id, val, shineVal);
        }
    }
    sort(scales.begin(), scales.end());
    cout << scales[0].id << endl;
}
```

## Part 3

### Solution 1: evaluating binary values, counting, map

```cpp
regex pattern(R"((\d+):([rR]+)\s+([gG]+)\s+([bB]+)\s+([sS]+))");

int eval(string s) {
    int res = 0, N = s.size();
    reverse(s.begin(), s.end());
    for (int i = 0; i < N; ++i) {
        if (isupper(s[i])) res += (1 << i);
    }
    return res;
}

string getColor(const string& red, const string& green, const string& blue) {
    int r = eval(red), g = eval(green), b = eval(blue);
    if (r > g && r > b) return "red";
    if (g > r && g > b) return "green";
    if (b > r && b > g) return "blue";
    return "none";
}

string getShine(const string& shine) {
    int s = eval(shine);
    if (s <= 30) return "matte";
    if (s >= 33) return "shiney";
    return "none";
}

const vector<string> colorKeys = {"red-matte", "red-shiny", "green-matte", "green-shiny", "blue-matte", "blue-shiny"};

void solve() {
    string line;
    int ans = 0, best = 0;
    map<string, int> colorCount, colorScore;
    while (getline(cin, line)) {
        smatch match;
        if (regex_match(line, match, pattern)) {
            int id = stoi(match[1]);
            string red = match[2], green = match[3], blue = match[4], shine = match[5];
            string color = getColor(red, green, blue);
            string shineType = getShine(shine);
            string key = color + "-" + shineType;
            colorCount[key]++;
            colorScore[key] += id;
        }
    }
    for (const string& key : colorKeys) {
        if (colorCount[key] > best) {
            best = colorCount[key];
            ans = colorScore[key];
        }
    }
    cout << ans << endl;
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

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```