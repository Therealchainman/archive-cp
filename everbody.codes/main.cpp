#include <iostream>
#include <pthread.h>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <deque>
#include <queue>
#include <numeric>
#include <stack>
#include <cassert>
#include <cstring>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <limits>
#include <functional>
#include <filesystem>
#include <fstream>
using namespace std;
#define endl '\n'
#define LOCAL

using int64 = int64_t;
using uint64 = unsigned long long;
using int128 = __int128_t;

#ifdef LOCAL
#include "src/debug.h"
#else
#define debug(...) 42
#endif


namespace fs = std::filesystem;

fs::path create_path(const std::string& directory, const std::string& file_name) {
    return fs::path(directory) / file_name;
}

string name = "everybody_codes_e2025_q01_p1.txt";

vector<string> process(const string& s, char delimiter = ' ') {
    vector<string> ans;
    istringstream iss(s);
    string word;
    while (getline(iss, word, delimiter)) ans.emplace_back(word);
    return ans;
}

void solve() {
    string S1, S2;
    getline(cin, S1);
    getline(cin, S2);
    getline(cin, S2);
    vector<string> A = process(S1, ','), B = process(S2, ',');
    int N = A.size();
    for (const string& s : B) {
        char dir = s[0];
        int x = stoi(s.substr(1));
        int j = 0;
        if (dir == 'R') {
            j = x % N;
        } else {
            j = -x;
            while (j < 0) j += N;
        }
        swap(A[0], A[j]);
    }
    cout << A[0] << endl;
}

signed main() {
    fs::path input_path = create_path("inputs", name);
    fs::path output_path = create_path("outputs", name);
    ifstream input_file(input_path);
    if (!input_file.is_open()) {
        std::cerr << "Error: Failed to open input file: " << input_path << endl;
        return 1;  // Exit with error if file cannot be opened
    }
    ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Failed to open output file: " << output_path << endl;
        return 1;  // Exit with error if file cannot be opened
    }
    cin.rdbuf(input_file.rdbuf());
    cout.rdbuf(output_file.rdbuf());
    solve();
    cin.rdbuf(nullptr);
    cout.rdbuf(nullptr);
    return 0;
}


/*
problem solve

You can output floats with using cout << fixed << setprecision(12) << p << endl;

This is to avoid stack overflow error on large recursion depths.

MACOS:
g++ -O2 main.cpp -o main
There is a hard limit on stack size, which is 64 MB. 
The only way to change the stack size is to run the following method to compile:
g++ -O2 "-Wl,-stack_size,20000000" main.cpp -o main

other helpful ulimit commands:
ulimit -a to see the limits
ulimit -Hs to see the hard limit of stack size
ulimit -Ss to see the soft limit of stack size

LINUX:
Needs to only be ran for each bash/shell
ulimit -s unlimited
This is the compiler code that will add some optimizations with the -O2
g++ -O2 main.cpp -o main
If you want to debug then consider running this.
g++ -g main.cpp -o main
then you can run gdb ./main
and type run into the terminal and enter debug session.
And use `bt` to get the backtrace

WINDOWS:
g++ "-Wl,--stack=26843546" main.cpp -o main
*/
