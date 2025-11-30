# Google Code Jam 2021

# Round 1A

## Append Sort

### Solution 1: 

```cpp
typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, N;
    string X;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> N;
        vector<string> nums;
        while (N--)
        {
            cin >> X;
            nums.push_back(X);
        }
        bool greater, lesser;
        int ans = 0;
        int prev, cur;
        for (int i = 1; i < nums.size(); i++)
        {
            prev = nums[i - 1].size(), cur = nums[i].size();
            if (cur > prev)
            {
                continue;
            }
            greater = false, lesser = false;
            for (int j = 0; j < cur; j++)
            {
                if (nums[i - 1][j] > nums[i][j])
                {
                    lesser = true;
                    break;
                }
                if (nums[i - 1][j] < nums[i][j])
                {
                    greater = true;
                    break;
                }
            }
            if (greater || lesser)
            {
                int diff = max(0, prev - cur);
                ans = greater ? ans + diff : ans + diff + 1;
                for (int j = 0; j < diff + lesser; j++)
                {
                    nums[i] += '0';
                }
            }
            else
            {
                for (int j = cur; j < prev; j++)
                {
                    nums[i] += nums[i - 1][j];
                }
                bool found = false;
                int idx = 0;
                for (int j = prev - 1; j >= cur; j--)
                {
                    if (nums[i][j] < 9 + '0')
                    {
                        nums[i][j]++;
                        found = true;
                        idx = j + 1;
                        break;
                    }
                }
                if (!found)
                {
                    for (int j = cur; j < prev; j++)
                    {
                        nums[i][j] = '0';
                    }
                    nums[i] += '0';
                    ans++;
                }
                else
                {
                    for (int j = idx; j < prev; j++)
                    {
                        nums[i][j] = '0';
                    }
                }
                ans += prev - cur;
            }
        }
        cout << "Case #" << t << ": " << ans << endl;
    }
}

```

## Hacked Exam

### Solution 1: 

```cpp
typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, N, Q, m, num, den;
    string s;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> N >> Q;
        vector<string> vec;
        vector<int> M;
        string ans = "";
        for (int i = 0; i < N; i++)
        {
            cin >> s >> m;
            vec.push_back(s);
            M.push_back(m);
        }
        if (N == 1)
        {
            den = 1;
            int q = vec[0].size();
            num = max(q - M[0], M[0]);
            for (int i = 0; i < vec[0].size(); i++)
            {
                if (M[0] < q - M[0] && vec[0][i] == 'F')
                {
                    ans += 'T';
                }
                else if (M[0] < q - M[0])
                {
                    ans += 'F';
                }
                else
                {
                    ans += vec[0][i];
                }
            }
        }
        else if (N == 2)
        {
            den = 1;
            int m1 = M[0], m2 = M[1], l1 = 0, l2 = 0;
            string s1 = vec[0], s2 = vec[1];
            int n = s1.size();
            for (int i = 0; i < n; i++)
            {
                if (s1[i] == s2[i])
                {
                    l1++;
                }
                else
                {
                    l2++;
                }
            }
            int x = (m1 + m2 - l2) / 2;
            int y = m2 - x;
            num = max(x, l1 - x) + max(y, l2 - y);
            for (int i = 0; i < n; i++)
            {
                if (s1[i] == s2[i] && l1 - x > x)
                {
                    if (s1[i] == 'F')
                    {
                        ans += 'T';
                    }
                    else
                    {
                        ans += 'F';
                    }
                }
                else if (s1[i] == s2[i])
                {
                    ans += s1[i];
                }
                else if (l2 - y > y)
                {
                    ans += s1[i];
                }
                else
                {
                    ans += s2[i];
                }
            }
        }
        else
        {
            //hahahah
            num = 1;
            den = 1;
        }
        cout << "Case #" << t << ": " << ans << ' ' << num << '/' << den << endl;
    }
}

```

## Prime Time

### Solution 1: 

```cpp
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

int LIMIT = 29940;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, M, P, curSum;
    ll sum, prod, maxEqual, N;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> M;
        maxEqual = 0, sum = 0;
        unordered_map<int, ll> primes;
        while (M--)
        {
            cin >> P >> N;
            sum += P * N;
            primes[P] += N;
        }
        for (int i = 2; i <= LIMIT; i++)
        {
            prod = sum - i;
            curSum = 0;
            if (prod < 2)
            {
                continue;
            }
            bool works = true;
            for (int j = 2; j <= 499 && prod > 1; j++)
            {
                int cnt = 0;
                while (prod % j == 0)
                {

                    prod /= j;
                    cnt++;
                    curSum += j;
                }
                if (cnt > primes[j])
                {
                    works = false;
                    break;
                }
            }
            if (prod == 1 && works && curSum == i)
            {
                maxEqual = sum - i;
                break;
            }
        }
        cout << "Case #" << t << ": " << maxEqual << endl;
    }
}
```

# Round 1B

## Broken Clock

### Solution 1: 

```cpp

ll M = 720LL * 1000LL * 1000LL * 1000LL;

bool check(ll h, ll m, ll s)
{

    if ((m == 12 * (h % (M * 5LL))) && (s == 60 * (m % M)))
    {
        return true;
    }
    return false;
}

void output(ll h, ll m, ll s)
{
    cout << h / (5 * M) << ' ' << m / M << ' ' << s / M << ' ' << (s % M) / 720 << endl;
}

bool find(ll h, ll m, ll s)
{
    for (int Q = 0; Q < 60; Q++)
    {

        ll shift = (60 * (M * Q - m) + s) / 59;
        ll H = mod(h + shift, 60 * M);
        ll min = mod(m + shift, 60 * M);
        ll S = mod(s + shift, 60 * M);

        if (check(H, min, S))
        {
            output(H, min, S);
            return true;
        }
    }
    return false;
}

typedef pair<int, int> p2;
int main()

{

    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T;
    ll A, B, C;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> A >> B >> C;
        vector<ll> ticks = {A, B, C};
        cout << "Case #" << t << ": ";
        do
        {
            if (find(ticks[0], ticks[1], ticks[2]))
            {
                break;
            }
        } while (next_permutation(ticks.begin(), ticks.end()));
    }
}
```

## Digit blocks

### Solution 1: 

```cpp
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

int find9(vector<int> &cnt, int in, int B)
{
    int maxx = -1;
    int maxxi = -1;
    for (int i = 0; i < cnt.size(); i++)
    {
        if (cnt[i] > maxx && cnt[i] < B)
        {
            maxx = cnt[i];
            maxxi = i;
        }
    }
    return maxxi;
}

int find8(vector<int> &cnt, int in, int B)
{
    int maxx = -1;
    int maxxi = -1;
    for (int i = 0; i < cnt.size(); i++)
    {
        if (cnt[i] > maxx && cnt[i] < B - 1)
        {
            maxx = cnt[i];
            maxxi = i;
        }
    }
    if (maxxi == -1)
    {
        for (int i = 0; i < cnt.size(); i++)
        {
            if (cnt[i] > maxx && cnt[i] < B)
            {
                maxx = cnt[i];
                maxxi = i;
            }
        }
    }
    return maxxi;
}

int find(vector<int> &cnt, int in, int B)
{
    int minn = B + 1;
    int minni = -1;
    for (int i = 0; i < cnt.size(); i++)
    {
        if (cnt[i] < minn)
        {
            minn = cnt[i];
            minni = i;
        }
    }
    return minni;
}

typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, D, B, N, cand;
    ll in;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> N >> B;
        vector<int> cnt(N, 0);
        for (int i = 0; i < N * B; i++)
        {
            cin >> in;
            if (in == 9)
            {
                cand = find9(cnt, in, B);
                cnt[cand]++;
                cout << cand << endl;
                continue;
            }
            if (in == 8)
            {
                cand = find8(cnt, in, B);
                cnt[cand]++;
                cout << cand << endl;
                continue;
            }
            cand = find(cnt, in, B);
            cnt[cand]++;
            cout << cand << endl;
        }
    }
}

```

## Subtransmutation

### Solution 1: 

```cpp
template <class T>
void output(int t, T out)
{
    cout << "Case #" << t << ": " << out << endl;
}

typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, N, A, B, u;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> N >> A >> B;
        unordered_map<int, int> U;
        int total = 0;
        for (int i = 1; i <= N; i++)
        {
            cin >> u;
            U[i] += u;
            total += u;
        }
        if (!isPossible(A, B, U))
        {
            output(t, IMPOSSIBLE);
            continue;
        }
        for (int i = 1; i < LIMIT; i++)
        {
            int cnt = 0;
            queue<int> q;
            q.push(i);
            unordered_map<int, int> NU = U;
            while (!q.empty())
            {
                int cur = q.front();
                q.pop();
                if (cur <= 0)
                {
                    continue;
                }
                if (NU[cur] > 0)
                {
                    NU[cur]--;
                    cnt++;
                }
                else
                {
                    q.push(cur - A);
                    q.push(cur - B);
                }
            }
            if (cnt == total)
            {
                output(t, i);
                break;
            }
        }
    }
}
```

# Round 1C

## Closest Pick

### Solution 1: 

```cpp
template <class T>
void output(int t, T out)
{
    cout << "Case #" << t << ": " << out << endl;
}

typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, N, K, p;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> N >> K;
        vector<int> tickets;
        int maxx = 0;
        for (int i = 0; i < N; i++)
        {
            cin >> p;
            tickets.push_back(p);
        }
        sort(tickets.begin(), tickets.end());
        vector<int> best;
        best.push_back(tickets[0] - 1);
        best.push_back(K - tickets.back());
        for (int i = 1; i < N; i++)
        {
            maxx = max(maxx, tickets[i] - tickets[i - 1] - 1);
            best.push_back((tickets[i] - tickets[i - 1]) / 2);
        }
        sort(best.rbegin(), best.rend());
        maxx = max(maxx, best[0] + best[1]);
        output(t, (double)maxx / (double)K);
    }
}
```

## Double or Noting

### Solution 1: 

```cpp
const string IMPOSSIBLE = "IMPOSSIBLE";

bool isPossible(string &S, string &E)
{
    int cntS = 0, cntE = 0;
    for (int i = 1; i < E.size(); i++)
    {
        if (E[i - 1] == '0' && E[i] == '1')
        {
            cntE++;
        }
    }
    if (cntE == 0)
    {
        return true;
    }
    for (int i = 1; i < S.size(); i++)
    {
        if (S[i - 1] == '0' && S[i] == '1')
        {
            cntS++;
        }
    }
    return cntS == cntE;
}

template <class T>
void output(int t, T out)
{
    cout << "Case #" << t << ": " << out << endl;
}

string notBit(string &s)
{
    string res = "";
    bool prev = false;
    for (int i = 0; i < s.size(); i++)
    {
        if (s[i] == '0')
        {
            prev = true;
            res += '1';
        }
        else if (prev)
        {
            res += '0';
        }
    }
    if (res.size() == 0)
    {
        res += '0';
    }
    return res;
}

string doubleVal(string &s)
{
    string res = "";
    if (s.size() == 1 && s[0] == '0')
    {
        res += '0';
        return res;
    }
    res += s + '0';
    return res;
}

const int LIMIT = 1e9;
typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, oper;
    string S, E, B;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> S >> E;
        queue<pair<string, int>> q;
        unordered_set<string> seen;
        q.emplace(S, 0);
        seen.insert(S);
        int cnt = 0;
        bool isPossible = false;
        while (!q.empty())
        {
            tie(B, oper) = q.front();
            cnt++;
            if (cnt == LIMIT)
            {
                break;
            }
            if (B == E)
            {
                output(t, oper);
                isPossible = true;
                break;
            }
            q.pop();
            if (B.size() > 32)
            {
                continue;
            }
            string n = notBit(B);
            if (seen.count(n) == 0)
            {
                q.emplace(n, oper + 1);
                seen.insert(n);
            }
            string d = doubleVal(B);
            if (seen.count(d) == 0)
            {

                q.emplace(d, oper + 1);
                seen.insert(d);
            }
        }
        if (!isPossible)
        {
            output(t, IMPOSSIBLE);
        }
    }
}
```

## Roaring Years

### Solution 1: 

```cpp
template <class T>
void output(int t, T out)
{
    cout << "Case #" << t << ": " << out << endl;
}

bool isValid(vector<int> &vec)
{
    for (int i = 1; i < vec.size(); i++)
    {
        if (vec[i] - vec[i - 1] != 1)
        {
            return false;
        }
    }
    return true;
}

bool find(int idx, string &Y, vector<int> vec)
{
    int n = Y.size();
    // printf("idx=%d\n", idx);
    // printVec(vec);
    // flush(cout);
    if (idx == n)
    {
        if (vec.size() < 2)
        {
            return false;
        }
        return isValid(vec) ? true : false;
    }
    for (int i = idx; i < n; i++)
    {
        if (i + 1 < n && Y[i + 1] == '0')
        {
            continue;
        }
        int nxt = stoi(Y.substr(idx, i - idx + 1));
        // printf("nxt=%d\n", nxt);
        vec.push_back(nxt);
        // printVec(vec);
        // flush(cout);
        if (find(i + 1, Y, vec))
        {
            return true;
        }
        vec.pop_back();
    }
    return false;
}

typedef pair<int, int> p2;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T, Y;
    cin >> T;
    for (int t = 1; t <= T; t++)
    {
        cin >> Y;
        vector<int> vec;
        for (int i = Y + 1;; i++)
        {
            string s = to_string(i);
            if (find(0, s, vec))
            {
                output(t, i);
                break;
            }
        }
    }
}

```