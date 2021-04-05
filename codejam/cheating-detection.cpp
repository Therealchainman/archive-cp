#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>
#include <regex>
#include <set>
#include <chrono>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <ctime>
#include <cassert>
#include <complex>
#include <string>
#include <cstring>
#include <chrono>
#include <random>
#include <array>
#include <bitset>
#define rep(i, n) for (i = 0; i < n; ++i)
#define REP(i, k, n) for (i = k; i <= n; ++i)
#define REPR(i, k, n) for (i = k; i >= n; --i)
#define pb push_back
#define all(a) a.begin(), a.end()
#define fastio               \
    ios::sync_with_stdio(0); \
    cin.tie(0);              \
    cout.tie(0)
#define ll long long
#define uint unsigned long long
#define inf 0x3f3f3f3f3f3f3f3f
#define mxl INT64_MAX
#define mnl INT64_MIN
#define mx INT32_MAX
#define mn INT32_MIN
#define endl '\n'
using namespace std;
using namespace std::chrono;

ll mod(ll a, ll b)
{
    return (a % b + b) % b;
}

template <class T>
void printVec(vector<T> vec)
{
    cout << "[";
    for (auto val : vec)
    {
        cout << val << ",";
    }
    cout << "]" << endl;
}
template <class T>
void print2DVec(vector<vector<T>> &M)
{
    cout << "[" << endl;
    for (vector<T> row : M)
    {
        cout << "[";
        for (auto val : row)
        {
            cout << val << ",";
        }
        cout << "]," << endl;
    }
    cout << "]" << endl;
}
void printArray(int arr[], int n)
{
    cout << "[";
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << ",";
    }
    cout << "]" << endl;
}
template <class T>
void printMap(T mp)
{
    cout << "{";
    for (auto &v : mp)
    {
        cout << "{" << v.first << "," << v.second << "},";
    }
    cout << "}" << endl;
}

template <class T>
void printUSet(unordered_set<T> st)
{
    cout << "{";
    for (auto v : st)
    {
        cout << v << ",";
    }
    cout << "}" << endl;
}

template <class T>
void printNameElem(string name, T a)
{
    cout << name << ": " << a << endl;
}

template <class T>
void printElem(T a)
{
    cout << a << endl;
}

template <class T1, class T2, class T3>
void printVec2Pair(vector<pair<pair<T1, T2>, T3>> p)
{
    cout << "{";
    for (auto v : p)
    {
        cout << "((" << v.first.first << "," << v.first.second << ")," << v.second << "),";
    }
    cout << "}" << endl;
}

template <class T1, class T2>
void printVecPair(vector<pair<T1, T2>> p)
{
    cout << "{";
    for (auto v : p)
    {
        cout << "(" << v.first << "," << v.second << "),";
    }
    cout << "}" << endl;
}

typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    // freopen("in/a.in", "r", stdin);
    int T, P;
    cin >> T >> P;
    int Q = 10000, N = 100;
    int cnt = 0;

    for (int t = 1; t <= T; t++)
    {
        vector<int> diff(Q);
        vector<double> total(N), hard(N);
        vector<string> answers;
        string answer;
        int cheater;
        for (int i = 0; i < N; i++)
        {
            cin >> answer;
            answers.push_back(answer);
            for (int j = 0; j < Q; j++)
            {
                if (answers[i][j] == '1')
                {
                    diff[j]++;
                }
            }
        }
        vector<p2> questions;
        for (int i = 0; i < Q; i++)
        {
            questions.emplace_back(diff[i], i);
        }

        for (int i = 0; i < Q; i++)
        {
            int cur = questions[i].second;
            for (int p = 0; p < N; p++)
            {
                if (answers[p][i] == '1')
                {
                    total[p]++;
                }
            }
        }
        sort(questions.begin(), questions.end());
        for (int i = 0; i < Q / 10; i++)
        {
            int qid = questions[i].second;
            for (int p = 0; p < N; p++)
            {
                if (answers[p][qid] == '1')
                {
                    hard[p] += (Q / 10) - i;
                }
            }
        }

        vector<pair<double, int>> players;
        for (int i = 0; i < N; i++)
        {
            players.emplace_back(total[i], i);
        }
        sort(players.begin(), players.end());
        double best = -1e9, score;
        vector<double> his(N);
        for (int i = 0; i < N; i++)
        {
            int me = players[i].second;
            int other = players[(i == 0 ? 1 : i - 1)].second;
            if (i != 0 && i != N - 1)
            {
                his[me] = hard[me] - (hard[other] + hard[players[i + 1].second]) / 2;
            }
            else
            {
                his[me] = hard[me] - hard[other];
            }
            if (his[me] > best)
            {
                best = his[me];
                cheater = players[i].second + 1;
            }
        }
        cout << "Case #" << t << ": " << cheater << endl;
    }
}