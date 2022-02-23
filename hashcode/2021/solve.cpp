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

int main()
{
    vector<const char *> inputs = {"inputs/a.txt", "inputs/b.txt", "inputs/c.txt", "inputs/d.txt", "inputs/e.txt", "inputs/f.txt"};
    vector<const char *> outputs = {"outputs/a.out", "outputs/b.out", "outputs/c.out", "outputs/d.out", "outputs/e.out", "outputs/f.out"};
    for (int i = 0; i < inputs.size(); i++)
    {
        freopen(inputs[i], "r", stdin);
        int D, I, S, V, F, B, E, L, p;
        string name;
        cin >> D >> I >> S >> V >> F;
        int maxL = 0;
        int minL = INT32_MAX;
        int maxP = 0;
        unordered_map<int, set<string>> inStreet;
        map<string, int> lmap;
        unordered_map<int, set<string>> outStreet;
        unordered_map<string, int> countP;
        while (S--)
        {
            cin >> B >> E >> name >> L;
            // not sure what will happen when say interStreet[E]= nothing, that is haven't initialized unordered_set
            if (inStreet.count(E) > 0)
            {
                inStreet[E].insert(name);
            }
            else
            {
                inStreet[E] = {name};
            }
            if (outStreet.count(B) > 0)
            {
                outStreet[B].insert(name);
            }
            else
            {
                outStreet[B] = {name};
            }
            lmap[name] = L;
            minL = min(minL, L);
            maxL = max(maxL, L);
        }
        cout << inputs[i] << endl;
        cout << minL << endl;
        cout << maxL << endl;
        while (V--)
        {
            cin >> p;
            while (p--)
            {
                cin >> name;
                countP[name]++;
                maxP = max(countP[name], maxP);
            }
        }
        cout << maxP << endl;
        ofstream file;
        file.open(outputs[i]);
        file << I << endl;
        for (int i = 0; i < I; i++)
        {
            file << i << endl;
            set<string> curOut = outStreet[i];
            set<string> cur = inStreet[i];
            file << cur.size() << endl;
            set<string>::iterator it;
            for (it = cur.begin(); it != cur.end(); it++)
            {
                if (lmap[*it] < 7 && countP[*it] > 400)
                {
                    file << *it << " " << 5 << endl;
                }
                else
                {
                    file << *it << " " << 1 << endl;
                }
            }
        }
        file.close();
    }
    return 0;
}

/*
Notes:
Idea to keep a function of the number of vehicles that pass through every intersection
That would have been a good idea, and to leave those lights green for longer
*/
