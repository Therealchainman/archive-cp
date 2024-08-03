# TeamsCode Summer 2024 Advanced Division

## B. Monkey Arrays

### Solution 1: 

```cpp
int N, X, Y, K;

void solve() {
    cin >> N >> X >> Y >> K;
    vector<int> arr(N);
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    int lx = -1, ly = -1, j = 0, ans = 0;
    for (int i = 0; i < N; i++) {
        if (arr[i] == K || arr[i] > X || arr[i] < Y) {
            lx = ly = -1;
            j = i + 1;
        } else if (arr[i] == X) {
            lx = i;
        } else if (arr[i] == Y) {
            ly = i;
        }
        int sz = min(lx, ly) - j + 1;
        ans += max(0LL, sz);
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## C. Monkey Math Tree

### Solution 1: 

```cpp

```
## D. Kawaii the Rinbot

### Solution 1:  polynomial rolling hashing, convert base 10 to base b = 95, use 4 characters to represent in base 95

part 1:  Find the P value that results in no hash collisions among any strings.

```cpp
const int C = 4, B = 95;
int M = pow(B, C);
string text;
vector<string> titles;

// try some p until it works. 
int polynomial_hash(const string &s, int p) {
    int h = 0;
    for (char c : s) {
        h = (h * p + c) % M;
    }
    return h;
}

void solve() {
    while (getline(cin, text)) titles.push_back(text);
    for (int p = 96; p < 3'000; p++) {
        bool ok = true;
        set<int> hashes;
        for (const string &text : titles) {
            int h = polynomial_hash(text, p);
            if (hashes.find(h) != hashes.end()) { // found collision
                ok = false;
                break;
            }
            hashes.insert(h);
        }
        if (ok) {
            cout << p << endl;
            break;
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    freopen("all_titles.txt", "r", stdin);
    freopen("test_p.txt", "w", stdout);
    solve();
    return 0;
}
```

part 1: offline to create string with all encoded 4 character hash representations.

```cpp
const int C = 4, B = 95, P = 96;
const set<char> escaped_chars = {'"', '?', '\\'};
const char ESC = '\\';
int M = pow(B, C);
string text, hashed_texts;
vector<string> titles;

// try some p until it works. 
int polynomial_hash(const string &s) {
    int h = 0;
    for (char c : s) {
        h = (h * P + c) % M;
    }
    return h;
}

string convert_base(int h, int b) {
    string s;
    for (int i = 0; i < 4; i++) {
        char c = (h % b) + 32;
        if (escaped_chars.find(c) != escaped_chars.end()) s += ESC;
        s += c;
        h /= b;
    }
    return s;
}

void solve() {
    hashed_texts = "";
    set<int> hashes;
    while (getline(cin, text)) {
        int h = polynomial_hash(text);
        assert(hashes.find(h) == hashes.end()); // no collisions
        hashed_texts += convert_base(h, B);
    }
    cout << hashed_texts << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    freopen("all_titles.txt", "r", stdin);
    freopen("all_titles_hashed.txt", "w", stdout);
    solve();
    return 0;
}
```

Part 2: The actual code to solve the problem
create a map to get lin number from each 4 character representation of strings.

```cpp
const string hashed_texts = [REDACTED]
const int C = 4, B = 95, P = 96;
int M = pow(B, C), T, MOD;
map<string, int> line_num;

// try some p until it works. 
int polynomial_hash(const string &s) {
    int h = 0;
    for (char c : s) {
        h = (h * P + c) % M;
    }
    return h;
}

string convert_base(int h, int b) {
    string s;
    for (int i = 0; i < 4; i++) {
        char c = (h % b) + 32;
        s += c;
        h /= b;
    }
    return s;
}

void solve() {
    for (int i = 0; i < hashed_texts.size(); i += 4) {
        string text = hashed_texts.substr(i, 4);
        line_num[text] = i / 4 + 1;
    }
    cin >> T >> MOD;
    cin.ignore();
    while (T--) {
        string text;
        getline(cin, text);
        string ht = convert_base(polynomial_hash(text), B);
        cout << line_num[ht] % MOD << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## E. Waymo orzorzorz

### Solution 1: 

```cpp

```

## F. Stage 4

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

## H. Thomas Sometimes Hides His Feelings in C++

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