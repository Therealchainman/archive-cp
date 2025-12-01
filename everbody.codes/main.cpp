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
#include <tuple>
#include <regex>
#include <utility>
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

string name = "a.txt";

void solve() {
    
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
    auto t0 = std::chrono::steady_clock::now();
    solve();
    auto t1 = std::chrono::steady_clock::now();
    cin.rdbuf(nullptr);
    cout.rdbuf(nullptr);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cerr << "Elapsed: " << ms << " ms" << endl;
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
