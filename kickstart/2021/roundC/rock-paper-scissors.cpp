#include <bits/stdc++.h>
using namespace std;
template <class T>
string toString(vector<T> &vec)
{
    stringstream res;
    copy(vec.begin(), vec.end(), ostream_iterator<T>(res, ""));
    return res.str().c_str();
}
double probability(int count, int round)
{
    return (double)count / (double)(round - 1);
}
int main()
{
    int T, X;
    cin >> T >> X;
    for (int t = 1; t <= T; t++)
    {
        int W, E;
        cin >> W >> E;
        vector<char> instructions;
        int p = 0, r = 0, s = 0; // number of times I've chosen rock,paper,scissors
        // double Pp, Pr, Ps;       // probability of opponent choice
        for (int i = 1; i <= 60; i++)
        {
            // Pp = probability(r, i);
            // Pr = probability(s, i);
            // Ps = probability(p, i);
            if (p == r && s == r)
            {
                instructions.push_back('R');
                r++;
            }
            else if (r > s)
            {
                instructions.push_back('S');
                s++;
            }
            else if (p < r && p < s)
            {
                instructions.push_back('P');
                p++;
            }
        }
        cout << "Case #" << t << ": " << toString(instructions) << endl;
    }
}
