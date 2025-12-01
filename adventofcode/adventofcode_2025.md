# Advent of Code 2025

## Day 1

### Part 1: modular arithmetic, circular indexing, 

```cpp
const int MOD = 100;

void solve() {
    int ans = 0;
    int pos = 50;
    string line;
    while (getline(cin, line)) {
        int step = stoi(line.substr(1));
        int r = step % MOD;
        pos = pos + (line[0] == 'R' ? r : -r);
        if (pos < 0) pos += MOD;
        pos %= MOD;
        if (pos == 0) ans++;
    }
    debug(ans, "\n");
}
```

### Part 2: modular arithmetic, remainders

1. step / MOD is the number of full circles you make so they visit 0. 
1. r is number of extra clicks that do not form a full circle. 
1. rotating left, you must check that you started above 0, pos > 0, and then you ended at or below 0, npos <= 0

```cpp
const int MOD = 100;

void solve() {
    int ans = 0;
    int pos = 50;
    string line;
    while (getline(cin, line)) {
        int step = stoi(line.substr(1));
        ans += step / MOD;
        int r = step % MOD;
        if (pos == 0 && line[0] == 'L') ans--; // edge case
        pos += (line[0] == 'R' ? r : -r);
        if (pos <= 0 || pos >= MOD) ans++;
        if (pos < 0) pos += MOD;
        if (pos >= MOD) pos -= MOD;
    }
    debug(ans, "\n");
}
```

## Day 2

### Part 1: 

```cpp

```

### Part 2: 

```cpp

```

## Day 3

### Part 1: 

```cpp

```

### Part 2: 

```cpp

```
