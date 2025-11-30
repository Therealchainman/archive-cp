# Advent of Code 2020

## Day 1

### Part 2: 

```cpp

```

## Day 2

### Part 2: 

```cpp

```

## Day 3

### Part 2: 

```cpp
typedef vector<int> vec;
typedef long long int lli;

lli countTrees(vector<string> map, int dx, int dy) {
    lli ans = 0;
    int C = map[0].size(), R = map.size();
    int c = dx;
    for (int r = dy;r<R;r+=dy) {
        if (map[r][c%C]=='#') {
            ans++;
        } 
        c+=dx;
    }
    return ans;
}
```

## Day 4

### Part 2: 

```cpp
bool isByr(string birth) {
    if (birth.size()!=4) {
        return false;
    }
    lli x = stoi(birth);
    return x>=1920 && x<=2002;
}

bool isIyr(string issue) {
    if (issue.size()!=4) {
        return false;
    }
    lli x = stoi(issue);
    return x>=2010 && x<=2020;
}

bool isEyr(string exp) {
    if (exp.size()!=4) {
        return false;
    }
    int x = stoi(exp);
    return x>=2020 && x<=2030;
}

bool isHgt(string he) {
    string unit = "";
    string val = "";
    for (int i = 0;i<he.size();i++) {
        if (isdigit(he[i])) {
            val+=he[i];
        } else {
            unit+=he[i];
        }
    }
    int x = stoi(val);
    if (unit == "cm" ) {
        return x>=150 && x<=193;
    }
    if (unit == "in") {
        return x>=59 && x<=76;
    }
    return false;
}

bool isHcl(string ha) {
    if (ha.size()!=7) {
        return false;
    }
    set<char> hexa = {'a','b','c','d','e','f'};
    for (int i = 1;i<ha.size();i++) {
        if (isdigit(ha[i])) {
            continue;
        }
        if (hexa.find(ha[i])!=hexa.end()) {
            continue;
        }
        return false;
    }
    return true;
}

bool isEcl(string eye) {
    set<string> eyes = {"amb","blu","brn","gry","grn","hzl","oth"};
    if (eyes.find(eye)!=eyes.end()) {
        return true;
    }
    return false;
}

bool isPid(string pass) {
    if (pass.size()!=9) {
        return false;
    }
    for (char x : pass) {
        if (!isdigit(x)) {
            return false;
        }
    }
    return true;
}

bool isValid(vector<string> passport) {
    map<string,int> fields;
    fields["ecl"]=0;
    fields["pid"]=0;
    fields["eyr"]=0;
    fields["hcl"]=0;
    fields["byr"]=0;
    fields["iyr"]=0;
    fields["cid"]=0;
    fields["hgt"]=0;
    for (string data : passport) {
        string field = data.substr(0,3);
        fields[field]++;
        string input = data.substr(4);
        cout<<field<<endl;
        cout<<input<<endl;
        if (field=="ecl" && !isEcl(input)) {
            return false;
        } 
        if (field=="pid" && !isPid(input)) {
            return false;
        }
        if (field=="eyr" && !isEyr(input)) {
            return false;
        }
        if (field=="hcl" && !isHcl(input)) {
            return false;
        } 
        if (field=="byr" && !isByr(input)) {
            return false;
        }
        if (field=="iyr" && !isIyr(input)) {
            return false;
        } 
        if (field=="hgt" && !isHgt(input)) {
            return false;
        }
    }
    map<string,int>::iterator it;
    for (it=fields.begin();it!=fields.end();it++) {
        if (it->first=="cid") {
            continue;
        } 
        if (it->second!=1) {
            return false;
        }
    }
    return true;
}

int main() {
    string a;
    lli res = 0;
    freopen("inputDay4.txt","r",stdin);
    string b = "";
    vector<string> A;
    while (getline(cin,a)) {
        if (a=="") {
            cout<<b<<endl;
            istringstream s(b);
            string tmp;
            while (s >> tmp) {
                A.push_back(tmp);
            }
            res+=isValid(A);
            A = {};
            b = "";
            continue;
        }
        b+=a + ' ';
    }
    istringstream s(b);
    string tmp;
    while (s >> tmp) {
        A.push_back(tmp);
    }
    res+=isValid(A);
    cout<<res<<endl;
    return 0;
}
```

## Day 5

### Part 2: 

```cpp

```

## Day 6

### Part 2: 

```cpp
int countYes(vector<string> A) {
    int ans = 0;
    int count[26];
    for (int i = 0;i<26;i++) {
        count[i]=0;
    }
    for (string str : A) {
        for (char x : str) {
            count[x-'a']++;
        }
    }
    for (int i = 0;i<26;i++) {
        if (count[i]==A.size()) {
            ans++;
        }
    }
    return ans;
}

int main() {
    int res = 0;
    freopen("inputDay6.txt","r",stdin);
    string x;
    vector<string> A;
    while (getline(cin,x)) {
        if (x=="") {
            res+=countYes(A);
            A = {};
        } else {
            A.push_back(x);
        }
    }
    res+=countYes(A);
    cout<<res<<endl;
    return 0;
}
```

## Day 7

### Part 2: 

```cpp
// counts the number of bags that can hold a bag of type bag
int countBags(map<string,map<string,int>> graph, string bag) {
    int count = 0;
    if (graph[bag].empty()) {
        return 0;
    } else {
        for (pair<string,int> pai : graph[bag]) {
            count += pai.second*(1+countBags(graph, pai.first));
        }
    }
    return count;
}


int main() { 
    int res = 0;
    string input;
    freopen("inputDay7.txt","r",stdin);
    map<string, map<string,int>> graph;
    vector<string> sent;
    while (getline(cin,input)) {
        istringstream s(input);
        string tmp, a,key,val;
        string int_num = "^0$|^[1-9][0-9]*$";
        regex pattern(int_num);
        key="";
        sent = {};
        int cnt;
        map<string,int> listOfBags;
        while (getline(s,tmp,' ')) {
            if (regex_match(tmp,pattern)) {
                cnt=stoi(tmp);
                continue;
            } else if (tmp=="contain") {
                continue;
            } else if (tmp=="no") {
                break;
            }
            sent.push_back(tmp);
            if (sent.size()==3) {
                for (int i = 0;i<2;i++) {
                    key+=sent[i];
                    key+=' ';
                }
            } else if (sent.size()%3==0) {
                val = "";
                for (int i = sent.size()-3;i<sent.size()-1;i++) {
                    val+=sent[i];
                    val+=' ';
                }
                listOfBags[val]=cnt;
            }
        }
        graph[key]=listOfBags;
    }
    string curBag= "shiny gold ";
    res = countBags(graph, curBag);
    cout<<res<<endl;

    return 0;
}
```

## Day 8

### Part 2: 

```cpp
int computeAcc(vector<pair<string,int>> actions, int swapIdx) {
    int ans = 0;
    int i = 0;
    set<int> vis;
    while (vis.find(i)==vis.end()) {
        vis.insert(i);
        pair<string,int> p = actions[i];
        if (i == swapIdx) {
            if (p.first=="nop") {
                i+=p.second;
            } else {
                i++;
            }
            continue;
        }
        if (p.first=="acc") {
            ans+=p.second;
            i++;
        } else if (p.first=="nop") {
            i++;
        } else {
            i+=p.second;
        }
    }
    return ans;
}

// Compute the index for the appropriate swap idx that is basically 
bool isLastIdx(vector<pair<string,int>> actions, int idx) {
    int i = 0;
    set<int> vis;
    bool firstSeen = false;
    while (vis.find(i)==vis.end()) {
        if (i==actions.size()) {
            return true;
        }
        vis.insert(i);
        pair<string,int> p = actions[i];
        if (i==idx && !firstSeen) {
            firstSeen = true;
            if (p.first=="nop") {
                i+=p.second;
            } else {
                i++;
            }
            continue;
        }
        if (p.first=="acc" || p.first=="nop") {
            i++;
        } else {
            i+=p.second;
        }
    }
    return i==actions.size();
}

vector<int> swapIndices(vector<pair<string,int>> actions, string action) {
    int i = 0;
    set<int> vis;
    vector<int> ret(0,0);
    while (vis.find(i)==vis.end()) {
        vis.insert(i);
        pair<string,int> p = actions[i];
        if (p.first=="nop") {
            if (action=="nop") {
                ret.push_back(i);
            }
            i++;

        } else if (p.first=="jmp") {
            if (action=="jmp") {
                ret.push_back(i);
            }
            i+=p.second;
        } else {
            i++;
        }
    }
    return ret;
}

int solve(vector<pair<string,int>> A, bool swap) {
    if (!swap) {
        return computeAcc(A, -1);
    } else {
        vector<int> swapIdxToJmp = swapIndices(A, "nop");
        vector<int> swapIdxToNop = swapIndices(A, "jmp");
        for (int ji : swapIdxToJmp) {
            if (isLastIdx(A, ji)) {
                return computeAcc(A, ji);
            }
        }
        for (int ni : swapIdxToNop) {
            if (isLastIdx(A, ni)) {
                return computeAcc(A, ni);
            }
        }
    }
    return 0;
}


int main() { 

    freopen("inputDay8.txt","r",stdin);
    string input;
    vector<pair<string,int>> A;
    while (getline(cin,input)) {
        string key = "";
        string val = "";
        bool flag = true;
        for (int i = 0;i<input.size();i++) {
            if (input[i]==' ') {
                flag=false;
            }
            if (flag) {
                key+=input[i];
            } else {
                val+=input[i];
            }
        } 
        int x;
        if (val[0]=='-') {
            x = -stoi(val.substr(1));
        } else {
            x = stoi(val.substr(1));
        }
        A.push_back({key,x});
    }
    int part1 = solve(A, false);
    int part2 = solve(A, true);
    cout<<part1<<endl;
    cout<<part2<<endl;
    return 0;
}
```

## Day 9

### Part 2: 

```cpp
bool check(vector<ll> prev, ll val, ll start) {
    set<ll> vis(prev.begin()+start,prev.end());
    for (int i = start;i<start+prev.size();i++) {
        if (vis.find(val-prev[i])!=vis.end() && prev[i]!=2*prev[i]) {
            return true;
        }
    }
    return false;
}

ll solve(vector<ll> arr, ll sz) {
    vector<ll> prev;
    for (int i = 0;i<sz;i++) {
        prev.push_back(arr[i]);
    }
    ll start = 0;
    for (int j = sz;j<arr.size();j++) {
        if (!check(prev, arr[j], start++)) {
            return arr[j];
        }
        prev.push_back(arr[j]);
    }
    return 0;
}

ll solve2(vector<ll> arr, ll goal) {
    ll curMin = INT64_MAX;
    ll curMax = INT64_MIN;
    ll sum = 0;
    set<ll> res;
    ll lo = 0, hi = 0;
    while (hi<arr.size() && sum!=goal) {
        sum += arr[hi];
        res.insert(arr[hi++]);
        while (lo<arr.size() && sum>goal) {
            sum-=arr[lo];
            res.erase(arr[lo++]);
        }
    }
    set<ll>::iterator it;
    for (it=res.begin();it!=res.end();it++) {
        curMin = min(curMin, *it);
        curMax = max(curMax, *it);
    }
    return curMin+curMax;
}

int main() { 
    freopen("inputDay9.txt","r",stdin);
    ll x;
    vector<ll> arr;
    while (cin >>x) {
        arr.push_back(x);
    }
    ll preAmbleSize = 25;
    ll res = solve(arr,preAmbleSize);
    cout<<solve2(arr, res)<<endl;
    return 0;
}
```

## Day 10

### Part 2: 

```cpp

```

## Day 11

### Part 2: 

```cpp

```

## Day 12

### Part 2: 

```cpp
ll mod (ll a, ll b) {
    return (a % b + b) %b;
}

map<char, pair<int,int>> dirs = {{'E', {1,0}}, {'S',{0,-1}},{'W',{-1,0}},{'N',{0,1}}};

ll wx = 10, wy = 1;

void updateWayPoint(ll delta, char d) {
        ll dx, dy;
        tie(dx,dy) = dirs[d];
        wx+=(dx*delta);
        wy+=(dy*delta);
}

void rotation(ll delta, char orientation) {
    ll i = mod(delta/90,4);
    ll nx, ny;
    int R, L;
    R = orientation == 'R' ? 1 : 0;
    L = orientation == 'L' ? 1 : 0;
    while (i>0) {
        nx = R*wy+-L*wy;
        ny = -R*wx+L*wx;
        wx = nx;
        wy = ny;
        i--;
    }
}

int main() {

    freopen("inputDay12.txt","r",stdin);
    ll x = 0, y = 0;
    string direction;
    while (cin>>direction) {
        char d = direction[0];
        ll delta = stoi(direction.substr(1));
        if (d == 'F') {
            x+=(wx*delta);
            y+=(wy*delta);
        } else if (d == 'R' || d == 'L') {
            rotation(delta, d); 
        } else {
            updateWayPoint(delta, d);
        }
    }
    ll res = abs(x) + abs(y);
    cout<<res<<endl;
    return 0;
}
```

## Day 13

### Part 2: 

```cpp
ll mod (ll a, ll b) {
    return (a % b + b) %b;
}

ll chineseRemainderTheoremNaive(vector<ll> times) {
    int k = times.size();
        for (ll i = times[0];;i+=times[0]) {
            int offset = 1;
            int j;
            for (j=1;j<k;j++) {
                ll bus = times[j];
                if (bus==-1) {
                    offset++;
                    continue;
                }
                if (i%bus!=bus-offset) {
                    break;
                }
                offset++;
            }
        if (j==k) {
            return i;
        }
    }
    return 0;
}

ll solve(vector<vector<ll>> times) {
    int k = times.size();
    int si = 1;
    ll stepSize = times[0][0];
    ll i = 0;
    while (si<k) {
        while (true) {
            i+=stepSize;
            if ((i+times[si][1])%times[si][0] == 0) {
                break;
            }
        }
        stepSize*=times[si][0];
        si++;
    }
    return i;
}

int main() {

    freopen("inputDay13.txt","r",stdin);
    string buses;
    cin >>buses;
    istringstream s(buses);
    string bus;
    vector<vector<ll>> times;
    int offset = -1;

    while (getline(s,bus,',')) {
        offset++;
        if (bus=="x") {
            continue;
        } 
        ll val = stoi(bus);
        times.push_back({val,offset});
    }
    cout<<solve(times)<<endl;
    return 0;
}
```

## Day 14

### Part 2: 

```cpp
ll mod (ll a, ll b) {
    return (a % b + b) %b;
}

map<ll,ll> memMap;
map<ll,ll> results;

ll sum(map<ll,ll> mp) {
    ll res = 0;
    map<ll,ll>::iterator it;
    for (it=mp.begin();it!=mp.end();it++) {
        res+=(it->second);
    }
    return res;
}

string decBin(ll decimal) {
    ll mask;
    string ans = "";
    for (ll i = 0;i<36;i++) {
        mask = 1LL << i;
        if ((decimal&mask) > 0) {
            ans+='1';
        } else {
            ans+='0';
        }
    }
    return ans;
}

string evaluate(string mask, string bits) {
    int n = mask.size();
    for (int i = 0;i<n;i++) {
        if (mask[i]=='X') {
            bits[i]='X';
        }
        if (mask[i]=='1') {
            bits[i]='1';
        }
    }
    return bits;
}

ll binDec(string binary) {
    ll ans = 0;
    for (int i = 0;i<36;i++) {
        ans +=((binary[i]-'0')*pow(2,i));
    }
    return ans;
}

void updateMap(ll value, string bits) {
    int countX = 0;
    for (char ch : bits) {
        if (ch=='X') {
            countX++;
        }
    }
    for (ll mask = 0;mask<(1LL<<countX);mask++) {
        ll k = 0;
        string tmpKey = "";
        for (int i = 0;i<36;i++) {
            if (bits[i]=='X') {
                if ((mask&(1LL<<k))>0) {
                    tmpKey+='1';
                } else {
                    tmpKey+='0';
                }
                k++;
            } else {
                tmpKey+=bits[i];
            }
        }
        ll newKey = binDec(tmpKey);
        cout<<newKey<<endl;
        results[newKey]=value;
    }
}

void solve(string mask, vector<ll> memories) {
    if (mask == "") {
        return;
    }
    for (ll key : memories) {
        string bitsKey = decBin(key);
        string res = evaluate(mask, bitsKey);
        updateMap(memMap[key],res);
    }
}

int main() {

    freopen("inputDay14.txt","r",stdin);
    string input,tmp;
    string mask = "";
    vector<ll> memories;
    while (getline(cin,input)) {
        istringstream s(input);
        vector<string> inputs;
        while (getline(s,tmp,' ')) {
            inputs.push_back(tmp);
        }
        if (inputs[0]=="mask") {
            solve(mask, memories);
            mask = inputs[2];
            reverse(mask.begin(),mask.end());
            memories.clear();
        } else {
            int n = inputs[0].size();
            string key;
            bool start = false;
            for (int i = 0;i<n;i++) {
                if (inputs[0][i]==']') {
                    continue;
                }
                if (inputs[0][i]=='[') {
                    start = true;
                } else if (start) {
                    key+=inputs[0][i];
                }
            }
            ll intKey = stoi(key);
            ll intVal = stoi(inputs[2]);
            memories.push_back(intKey);
            memMap[intKey]=intVal;
        }
    }
    solve(mask,memories);
    cout<<sum(results)<<endl;
    return 0;
}

```

## Day 15

### Part 2: 

```cpp
ll mod (ll a, ll b) {
    return (a % b + b) %b;
}


int solve(vector<int> nums, int end) {
    int n = nums.size();
    unordered_map<int,int> posNum;
    int curNum;
    int i;
    for (i = 0;i<n;i++) {
        posNum[nums[i]]=i;
        curNum=nums[i];
    }
    int nextNum = 0;
    int count[10];
    while (i<end) {
        curNum=nextNum;
        if (posNum.find(nextNum)==posNum.end()) {\
            nextNum = 0;
            posNum[curNum]=i;
        } else {
            int prevIndex = posNum[curNum];
            int curIndex = i;
            posNum[curNum]=curIndex;
            nextNum=curIndex-prevIndex;
        }
        i++;
    }
    return curNum;
}

int main() {

    freopen("big.txt","r",stdin);
    string input,tmp;
    vector<int> starting_nums;
    while (getline(cin,input)) {
        istringstream s(input);
        vector<string> inputs;
        while (getline(s,tmp,',')) {
            int x = stoi(tmp);
            starting_nums.push_back(x);
        }
    }
    auto start = high_resolution_clock::now();
    int res = solve(starting_nums,30000000);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop-start);
    cout<<duration.count()<<endl;
    cout<<res<<endl;
    return 0;
}

```

## Day 16

### Part 2: 

```cpp
// Part 1
int solve(vector<int> values, vector<pair<int,int>> ranges) {
    int errSum = 0;
    int i = 0;
    int n = ranges.size();
    for (int v : values) {
        while (i<n && v>ranges[i].second) {
            i++;
        }
        if (v<ranges[i].first || v>ranges[i].second) {
            errSum += v;
        } 
    }
    return errSum;
}

vector<pair<int,int>> mergeRanges(vector<pair<int,int>> ranges) {
    vector<pair<int,int>> events;
    for (pair<int,int> r : ranges) {
        events.emplace_back(r.first,-1);
        events.emplace_back(r.second,1);
    }
    sort(events.begin(),events.end());
    vector<pair<int,int>> res;
    int count = 0;
    int start;
    for (pair<int,int> event : events) {
        int e, delta;
        tie(e,delta)=event;
        delta=-delta;
        count+=delta;
        if (count==1 && delta==1) {
            start = e;
        } else if (count==0 && delta==-1) {
            res.emplace_back(start,e);
        }
    }
    return res;
}

// int main() {

//     freopen("big.txt","r",stdin);
//     string input,tmp;
//     vector<pair<int,int>> ranges;
//     vector<int> values;
//     while (getline(cin,input)) {
//         if (input == "your ticket:") {
//             break;
//         }
//         istringstream s(input);
//         int j = 0;
//         while (getline(s,tmp,' ')) {
//             int i = 0;
//             if (!isdigit(tmp[i])) {
//                 continue;
//             }
//             string startstr = "";
//             while (isdigit(tmp[i])) {
//                 startstr+=tmp[i++];
//             }
//             int start = stoi(startstr);
//             i++;
//             string endstr = "";
//             while (isdigit(tmp[i])) {
//                 endstr+=tmp[i++];
//             }
//             int end = stoi(endstr);
//             ranges.emplace_back(start,end);
//         }
//     }
//     for (int i = 0;i<3;i++) {
//         getline(cin,input);
//     }
//     while (getline(cin,input)) {
//         istringstream s(input);
//         while (getline(s,tmp,',')) {
//             int x = stoi(tmp);
//             values.push_back(x);
//         }
//     }
//     sort(values.begin(),values.end());
//     // auto start = high_resolution_clock::now();
//     ranges = mergeRanges(ranges);
//     int res = solve(values, ranges);
//     // auto stop = high_resolution_clock::now();
//     // auto duration = duration_cast<seconds>(stop-start);
//     // cout<<duration.count()<<endl;
//     cout<<res<<endl;
//     return 0;
// }

// part 2
struct rule {
    string name;
    bool isDeparture;
    pair<int,int> bounds1;
    pair<int,int> bounds2;
    
    explicit rule(const string &line) {
        name = line.substr(0,line.find(':'));
        isDeparture = name.find("departure") == 0;
        string r1 = line.substr(name.size()+2,line.find(" or "));
        bounds1.first = stoi(r1.substr(0,r1.find('-')));
        bounds1.second = stoi(r1.substr(r1.find('-')+1));
        string r2 = line.substr(line.find(" or ")+4);
        bounds2.first = stoi(r2.substr(0,r2.find('-')));
        bounds2.second = stoi(r2.substr(r2.find('-')+1));
    }

    bool valueIsValid(int num) {
        return (num>=bounds1.first && num<=bounds1.second) || (num>=bounds2.first && num<=bounds2.second);
    }
};

struct puzzleInput {
    vector<rule> rules;
    vector<int> myTicket;
    vector<vector<int>> nearByTickets;
};

ll solve(puzzleInput p) {
    p.nearByTickets.erase(remove_if(p.nearByTickets.begin(),p.nearByTickets.end(), 
    [p](vector<int> tickets) { 
        return !all_of(tickets.begin(),tickets.end(), 
    [p](int ticketNum) {
        return any_of(p.rules.begin(),p.rules.end(),
    [ticketNum] (rule r) {
        return r.valueIsValid(ticketNum);
    });
    });
    })
    , p.nearByTickets.end());

    map<int, set<rule*>> rulesByCols;
    bool valid;
    for (int c = 0;c<p.myTicket.size();c++) {
        for (rule& r : p.rules) {
            valid = true;
            for (vector<int> curTicket : p.nearByTickets) {
                bool ticketValid = r.valueIsValid(curTicket[c]);
                valid &=ticketValid;
                if (!valid) {
                    break;
                }
            }
            if (valid) {
                rulesByCols[c].insert(&r);
            }
        }
    }
    set<rule*> usedRules;
    while (true) {
        auto it = find_if(rulesByCols.begin(),rulesByCols.end(),
        [&usedRules](pair<const int, set<rule*>> &rule) {
            return rule.second.size()==1 && !usedRules.count(*rule.second.begin());;
        });
        if (it==rulesByCols.end()) {
            break;
        }
        rule* curRule = *it->second.begin();
        usedRules.insert(curRule);
        for_each(rulesByCols.begin(),rulesByCols.end(), 
        [curRule] (pair<const int,set<rule*>> &rs) {
            if (rs.second.size()>1) {
                rs.second.erase(curRule);
            }
        });
    }

    ll res = 1;
    for (int i = 0;i<p.myTicket.size();i++) {
        if (rulesByCols[i].size()==1 && (*rulesByCols[i].begin())->isDeparture) {
            res*=p.myTicket[i];
        }
    }
    return res;
}


int main() {

    freopen("inputs/big.txt","r",stdin);
    string input,tmp;
    puzzleInput pInput;
    int section = 0, ticketNum;
    vector<int> nearTicket;
    while (getline(cin,input)) {
        if (input.empty()) {
            section++;
            getline(cin,input);
            continue;
        }
        if (section==0) {
            rule r = rule(input);
            pInput.rules.push_back(r);
        } else if (section==1) {
            istringstream s(input);
            while (getline(s,tmp,',')) {
                ticketNum = stoi(tmp);
                pInput.myTicket.push_back(ticketNum);
            }
        } else {
            nearTicket.clear();
            istringstream s(input);
            while (getline(s,tmp,',')) {
                ticketNum = stoi(tmp);
                nearTicket.push_back(ticketNum);
            }
            pInput.nearByTickets.push_back(nearTicket);
        }
    }
    ll res = solve(pInput);
    // auto start = high_resolution_clock::now();
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<seconds>(stop-start);
    // cout<<duration.count()<<endl;
    cout<<res<<endl;
    return 0;
}
```

## Day 17

### Part 2: 

```cpp
vector<string> inputVec;

ll solve(bool isW) {
    int minX = 0;
    int minY = 0;
    int maxZ = 0;
    int maxX = inputVec[0].size()-1;
    int maxY = inputVec.size()-1;
    int maxW = 0;
    map<p4,bool> space;
    map<p4,bool> nextSpace;
    vector<p4> neighbors;
    for (int x = 0;x<=maxX;x++) {
        for (int y = 0;y<=maxY;y++) {
            p4 pos = p4(x,y,0,0);
            space[pos] = (inputVec[x][y]=='#');
        }
    }
    for (int i = -1;i<=1;i++) {
        for (int j = -1;j<=1;j++) {
            for (int k = -1;k<=1;k++) {
                for (int w = -1*isW;w<=isW;w++) {
                    if (i==0 && j==0 && k==0 && w==0) {
                        continue;
                    }
                    p4 neigh = p4(i,j,k,w);
                    neighbors.push_back(neigh);
                }
            }
        }
    }
    for (int cycle = 0;cycle<6;cycle++) {
        for (int x = minX-1;x<=maxX+1;x++) {
            for (int y = minY-1;y <= maxY + 1; y++) {
                for (int z = 0; z <= maxZ + 1; z++) {
                    for (int w = 0; w <= maxW + 1; w++) {
                        int countActiveNeighbors = 0;
                        p4 pos = p4(x,y,z,w);
                        for (p4 delta : neighbors) {
                            p4 neigh = pos + delta;
                            neigh.z = abs(neigh.z);
                            neigh.w = abs(neigh.w);
                            countActiveNeighbors+=space[neigh];
                        }
                        if (space[pos]) {
                            if (countActiveNeighbors==2 || countActiveNeighbors==3) {
                                nextSpace[pos]=space[pos];
                            } else {
                                nextSpace[pos]=false;
                            }
                        } else {
                            if (countActiveNeighbors==3) {
                                nextSpace[pos]=true;
                            } else {
                                nextSpace[pos]=space[pos];
                            }
                        }
                    }
                }
            }
        }
        maxX++;
        minX--;
        maxY++;
        minY--;
        maxZ++;
        maxW+=isW;
        swap(space,nextSpace);
    }
    

    ll countActive = 0;
    for (pair<p4,bool> elem : space) {
        ll val = elem.second;
        if (elem.first.z>0) {
            val*=2;
        }
        if (elem.first.w>0) {
            val*=2;
        }
        countActive+=val;
    }
    return countActive;
}

int main() {
    freopen("inputs/inputDay17.txt","r",stdin);
    string input;
    while (getline(cin,input)) {
        inputVec.push_back(input);
    }
    // part 1
    cout<<solve(false)<<endl;
    auto start = high_resolution_clock::now();

    // part 2
    cout<<solve(true)<<endl;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);
    cout<<duration.count()<<endl;
    return 0;
}
```

## Day 18

### Part 2: 

```cpp
ll computeRow(string expr, bool prec) {
    stack<char> operatorStack;
    stack<ll> operandStack;
    ll a, b;
    for (char oper : expr) {
        if (isdigit(oper)) {
            operandStack.push(oper-'0');
        } else if (oper=='(') {
            operatorStack.push('(');
        } else if (oper==')') {
            while (!operatorStack.empty() && operatorStack.top()!='(') {
                a = operandStack.top();
                operandStack.pop();
                b = operandStack.top();
                operandStack.pop();
                operandStack.push(evaluate(a,b,operatorStack.top()));
                operatorStack.pop();
            }
            operatorStack.pop();
        } else if (oper == '+' || oper=='*') {
            while (!operatorStack.empty() && operatorStack.top() != '(' && (!prec || prec && oper=='*' && operatorStack.top()=='+')) {
                a = operandStack.top();
                operandStack.pop();
                b = operandStack.top();
                operandStack.pop();
                operandStack.push(evaluate(a,b,operatorStack.top()));
                operatorStack.pop();
            }
            operatorStack.push(oper);
        }
    }
    return operandStack.top();
}


ll solve(number n,bool prec) {
    ll ans = 0;
    for (string line : n.lines) {
        ans+=computeRow(line,prec);
    }
    return ans;
}

int main() {
    freopen("inputs/big.txt","r",stdin);
    number n;
    string input,tmp,operands;
    while (getline(cin,input)) {
        istringstream s(input);
        operands = "(";
        while (getline(s,tmp,' ')) {
            operands+=tmp;
        }
        operands+=")";
        n.lines.push_back(operands);
    }
    int start = 0;
    // part 1
    cout<<solve(n,false)<<endl;
    // part 2
    cout<<solve(n,true)<<endl;
    return 0;
}
```

## Day 19

### Part 2: 

```cpp
string trim(string s) {
    string ans = "";
    for (char ch : s) {
        if (ch=='"' || ch=='"') {
            continue;
        }
        ans+=ch;
    }
    return ans;
}

bool isNumber(const string &s) {
  return !s.empty() && all_of(s.begin(), s.end(), ::isdigit);
}

struct rule {
    string nameS, subR1, subR2;
    int numSubs = 1;
    int name;
    size_t sizeMax = numeric_limits<size_t>::max();
    bool isChar = false;
    string character;
    vector<string> R1, R2;

    explicit rule(string &line) {
        nameS = line.substr(0,line.find(':'));
        name = stoi(nameS);
        // start part 2
        if (name==8) {
            line = "8: 42 | 42 8";
        } else if (name==11) {
            line = "11: 42 31 | 42 11 31";
        }
        // end part 2
        size_t foundChar = line.find('"');
        if (foundChar<sizeMax) {
            isChar = true;
            character = trim(line.substr(foundChar));
            R1.push_back(character);
            return;
        }
        size_t found = line.find(" |");
        if (found<sizeMax) {
            subR1 = line.substr(nameS.size()+2,found-nameS.size()-2);
            subR2 = line.substr(found + 3);
            numSubs++;
        } else {
            subR1 = line.substr(nameS.size()+2);
        }
        istringstream s(subR1);
        string tmp;
        while (getline(s,tmp,' ')) {
            R1.push_back(tmp);
        }
        istringstream t(subR2);
        while (getline(t,tmp,' ')) {
            R2.push_back(tmp);
        }
    }
};

struct message {
    vector<string> texts;
    unordered_map<int, rule*> rules;
};

bool isTerminal(rule* rule) {
    return rule->isChar;
}

int solve(message m) {
    int ans = 0;
    for (string msg : m.texts) {
        vector<vector<string>> patterns(1, vector<string>(1,"0"));
        for (int i = 0;i<msg.size();i++) {
            vector<vector<string>> nextPatterns;
            for (int k = 0;k<patterns.size();k++) {
                vector<string> pattern = patterns[k];
                if (pattern.size()<=i) {
                    continue;
                }
                while (isNumber(pattern[i])) {
                    if (pattern.size()>msg.size()) {
                        break;
                    }
                    int rKey = stoi(pattern[i]);
                    rule* cRule = m.rules[rKey];
                    if (isTerminal(cRule)) {
                        pattern[i]=cRule->character;
                        continue;
                    }
                    // simple rule
                    if (cRule->subR2.empty()) {
                        for (int j = 0;j<cRule->R1.size();j++) {
                            string nextRuleId = cRule->R1[j];
                            if (j==0) {
                                pattern[i]=nextRuleId;
                                continue;
                            }
                            pattern.insert(pattern.begin()+i+j,nextRuleId);
                        }
                    } else {
                        vector<string> cPattern = pattern;
                        for (int j = 0;j<cRule->R1.size();j++) {
                            string nextRuleId = cRule->R1[j];
                            if (j==0) {
                                pattern[i]=nextRuleId;
                                continue;
                            }
                            pattern.insert(pattern.begin()+i+j,nextRuleId);
                        }
                        for (int j = 0;j<cRule->R2.size();j++) {
                            string nextRuleId = cRule->R2[j];
                            if (j==0) {
                                cPattern[i]=nextRuleId;
                                continue;
                            }
                            cPattern.insert(cPattern.begin()+j+i,nextRuleId);
                        }
                        patterns.push_back(cPattern);
                    }
                }
                string s;
                s.push_back(msg[i]);
                if (pattern[i]==s && pattern.size()<=msg.size()) {
                    nextPatterns.push_back(pattern);
                }
            }
            patterns = nextPatterns;
        }
        ans+=patterns.size();
    }
    return ans;
}

int main() {
    freopen("inputs/big.txt","r",stdin);
    string input;
    message m;
    while (getline(cin,input)) {
        if (input.empty()) {
            break;
        }
        rule* r = new rule(input);
        m.rules[r->name] = r;
    }
    while (getline(cin,input)) {
        m.texts.push_back(input);
    }
    // part 1

    // part 2
    auto start = high_resolution_clock::now();
    cout<<solve(m)<<endl;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);
    cout<<duration.count()<<endl;


    return 0;
}
```

## Day 20

### Part 2: 

```cpp
struct Tile {
    Tile(const vector<string> &tileInput) {
        regex idrx("Tile (\\d+)\\:");
        smatch match;
        regex_match(tileInput.front(), match, idrx);
        id = stoull(match[1].str().data());
        copy(tileInput.begin()+1,tileInput.end(),back_inserter(orig));
        rows = orig;
    }

    void setConfig(int i) {
        rows = orig;
        switch(i) {
            case 0 : flipHorizontal(); flipVertical(); break;
            case 1 : flipHorizontal(); rotate();break;
            case 2 : flipVertical(); rotate(); break;
            case 3 : rotate(); break;
            case 4 : flipHorizontal(); break;
            case 5 : flipVertical(); break;
            case 6 : break;
            case 7 : flipHorizontal(); flipVertical(); rotate(); break;
            default : cout<< "bad config requested"; break;
        }
    }

    string left() const {
        string s;
        for (auto line : rows) {
            s+=line.front();
        }
        return s;
    }

    string right() const {
        string s;
        for (auto line : rows) {
            s+=line.back();
        }
        return s;
    }

    string top() const {
        return rows.front();
    }

    string bottom() const {
        return rows.back();
    }

    void flipHorizontal() {
        for (auto &edge : rows) {
            reverse(edge.begin(),edge.end());
        }
    }

    void flipVertical() {
        reverse(rows.begin(), rows.end());
    }

    void rotate() {
        for (ll n = 0;n < rows.size()-1;n++) {
            for (ll m = n+1;m<rows.size();m++) {
                swap(rows[n][m],rows[m][n]);
            }
            reverse(rows[n].begin(),rows[n].end());
        }
        reverse(rows.back().begin(),rows.back().end());
    }
    string toString() const {
        string s;
        for (auto line : rows ) {
            s+=line + "\n";
        }
        return s;
    }

    ll id;
    vector<string> rows;
    vector<string> orig;
    Tile* south{nullptr};
    Tile* east{nullptr};
};

ll solve(vector<Tile> tiles) {
    queue<Tile*> tileQ;
    map<ll, int> goodIds;
    goodIds[tiles[0].id] = 0;
    tileQ.push(&tiles[0]);
    map<int, int> right, left, up, down;

    while (tileQ.size() > 0) {
        Tile* target = tileQ.front();
        target->setConfig(goodIds[target->id]);
        for (ll i = 0;i<tiles.size();i++) {
            if (goodIds.find(tiles[i].id) != goodIds.end()) {
                continue;
            }
            for (int j = 0;j<8;j++) {
                tiles[i].setConfig(j);
                if (target->right() == tiles[i].left()) {
                    goodIds[tiles[i].id]=j;
                    tileQ.push(&tiles[i]);
                } else if (target->left()==tiles[i].right()) {
                    goodIds[tiles[i].id]=j;
                    tileQ.push(&tiles[i]);
                } else if (target->top()==tiles[i].bottom()) {
                    goodIds[tiles[i].id]=j;
                    tileQ.push(&tiles[i]);
                } else if (target->bottom()==tiles[i].top()) {
                    tileQ.push(&tiles[i]);
                    goodIds[tiles[i].id]=j;
                }
            }
        }
        tileQ.pop();
    }

    // start part 1

    Tile *leftTile = nullptr;
    ll res = 1;
    for (Tile &tile1 : tiles) {
        int connections = 0;
        for (Tile &tile2 : tiles) {
            if (&tile1 != &tile2) {
                if (tile1.left() == tile2.right()) {
                    connections++;
                    tile2.east=&tile1;
                }
                if (tile1.top()==tile2.bottom()) {
                    connections++;
                    tile2.south=&tile1;
                }
                if (tile1.right()==tile2.left()) {
                    connections++;
                    tile1.east=&tile2;
                }
                if (tile1.bottom()==tile2.top()) {
                    connections++;
                    tile1.south=&tile2;
                }
            }
        }
        if (connections==2) {
                res *= tile1.id;
                if (tile1.east && tile1.south) {
                    leftTile = &tile1;
                }
            }
    }
    cout<<"part one: "<<res<<endl;
    // End part 1
    // start part 2
    vector<string> photo = {"Tile 0000:"};
    while (leftTile) {
        ll photoLine = photo.size();
        for (ll i = 0;i<tiles[0].rows.size()-2;i++) {
            photo.push_back("");
        }
        Tile *curTile = leftTile;
        while (curTile) {
            for (ll i = photoLine; i<photoLine+tiles[0].rows.size()-2;i++) {
                photo[i]+=curTile->rows[i-photoLine+1].substr(1,curTile->rows.size()-2);
            }
            curTile = curTile->east;
        }
        leftTile = leftTile->south;
    }
    Tile final(photo);
    for (int config = 0;config<8;config++) {
        final.setConfig(config);
        string monster1 = "                  # ";
        string monster2 = "#    ##    ##    ###";
        string monster3 = " #  #  #  #  #  #   ";

        int count = 0;
        for (ll i = 0;i<final.rows.size()-3;i++) {
            for (ll j = 0;j<final.rows.size()-monster1.size();j++) {
                bool ok = true;
                for (ll k = 0;k<monster1.size();k++) {
                    if ((monster1[k]=='#' && final.rows[i][k+j]!='#') 
                    || (monster2[k]=='#' && final.rows[i+1][k+j]!='#') 
                    || (monster3[k]=='#' && final.rows[i+2][k+j]!='#')) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    count++;
                }
            }
        }
        if (count>0) {
            ll rough = 0;
            for (ll i = 0;i<final.rows.size();i++) {
                for (ll j = 0;j<final.rows.size();j++) {
                    if (final.rows[i][j] == '#') {
                        rough++;
                    }
                }
            }
            cout<< "part two: " << rough-count*15<<endl;
        }
    }
    //end part 2

}

int main() {
    auto start = high_resolution_clock::now();

    string line;
    vector<string> lines;
    vector<Tile> tiles;
    freopen("inputs/inputDay20.txt","r",stdin);
    while (getline(cin,line)) {
        if (line.empty()) {
            tiles.push_back({lines});
            lines.clear();
        } else {
            lines.push_back(line);
        }
    }
    if (lines.size() > 0) {
        tiles.push_back({lines});
    }
    
    // Part 1 & 2
    solve(tiles);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);
    cout<<duration.count()<<endl;
}
```

## Day 21

### Part 2: 

```cpp
int main() {
    freopen("inputs/big.txt","r",stdin);
    map<string, set<string>> allergensToIngredients;
    multiset<string> allIngredients;
    string curLine;
    while (getline(cin,curLine)) {
        stringstream lineStream(curLine);
        set<string> curIngredients;
        set<string> curAllergens;
        string curToken;
        while (lineStream >> curToken) {
            if (curToken[0] == '(') {
                break;
            } else {
                curIngredients.insert(curToken);
            }
        }
        allIngredients.insert(curIngredients.begin(),curIngredients.end());
        while (lineStream >> curToken) {
            const auto curAllergen = curToken.substr(0,curToken.size()-1);
            curAllergens.insert(curAllergen);
            if (allergensToIngredients.count(curAllergen)) {
                auto& otherIngredients = allergensToIngredients.at(curAllergen);
                set<string> intersection;
                set_intersection(otherIngredients.cbegin(),otherIngredients.cend(), curIngredients.cbegin(), curIngredients.cend(), inserter(intersection, intersection.begin()));
                otherIngredients = move(intersection);
            } else {
                allergensToIngredients.insert({curAllergen, curIngredients});
            }
        }
    }
    while (any_of(allergensToIngredients.cbegin(),allergensToIngredients.cend(), [] (auto& it) {
        return it.second.size() > 1;})
    )  {
        for (const auto& allergenToIngredientA : allergensToIngredients) {
            if (allergenToIngredientA.second.size()==1) {
                const auto& toRemove = *allergenToIngredientA.second.begin();
                for (auto& allergenToIngredientB : allergensToIngredients) {
                    if (allergenToIngredientA.first == allergenToIngredientB.first) {
                        continue;
                    }
                    allergenToIngredientB.second.erase(toRemove);
                }
            }
        }
    }
    uintmax_t Part1{};
    {
        multiset<string> safeIngredients = allIngredients;
        for (const auto& allergicIngredient : allergensToIngredients) {
            safeIngredients.erase(*allergicIngredient.second.begin());
        } 
        Part1 = safeIngredients.size();
    }
    cout<< Part1<<endl;
    string Part2;
    {
        for (const auto& allergenToIngredient : allergensToIngredients) {
            Part2 += *allergenToIngredient.second.begin() + ',';
        }
        Part2.pop_back();
    }
    cout<<Part2<<endl;
}
```

## Day 22

### Part 2: 

```cpp
map<pair<deque<ll>,deque<ll>>, Player> memo;

ll score(Game g) {
    ll ans = 0;
    ll x = g.winner.deck.cards.size();
    while (x>0) {
        ans += (g.winner.deck.cards.front()*(x--));
        g.winner.deck.cards.pop_front();
    }
    return ans;
}

Player playGame(Game g, map<pair<deque<ll>,deque<ll>>,int> visitedDeck) {
    Player p1 = g.players[0];
    Player p2 = g.players[1];
    ll card1, card2;
    Player winner;
    pair<deque<ll>,deque<ll>> p;
    while (!p1.deck.cards.empty() && !p2.deck.cards.empty()) {
        p.first=p1.deck.cards;
        p.second=p2.deck.cards;
        if (visitedDeck[p]>0) {
            return p1;
        }
        visitedDeck[p]=1;
        card1 = p1.deck.cards.front();
        card2 = p2.deck.cards.front();
        p1.deck.cards.pop_front();
        p2.deck.cards.pop_front();
        if (p1.deck.cards.size()>=card1 && p2.deck.cards.size()>=card2) {
            Game g2;
            Player p3, p4;
            for (int i = 0 ; i < card1; i++) {
                p3.deck.cards.push_back(p1.deck.cards[i]);
            }
            for (int j = 0 ; j < card2; j++) {
                p4.deck.cards.push_back(p2.deck.cards[j]);
            }
            p3.id=p1.id;
            p4.id=p2.id;
            g2.players.push_back(p3);
            g2.players.push_back(p4);
            winner = playGame(g2,{});
        } else {
            if (card1>card2) {
                winner= p1;
            } else {
                winner= p2;
            }
        }
        if (winner.id==p1.id) {
            p1.deck.cards.push_back(card1);  
            p1.deck.cards.push_back(card2);
        } else {
            p2.deck.cards.push_back(card2);
            p2.deck.cards.push_back(card1);
        }
    }
    if (p1.deck.cards.empty()) {
        winner = p2;
    } else {
        winner = p1;
    }
    return winner;
}

int main() {
    auto start = high_resolution_clock::now();
    freopen("inputs/big.txt","r",stdin);
    string line;
    deque<ll> cards;
    int i = 0, player;
    Game g;
    Player p;
    Deck d;
    while (getline(cin,line)) {
        if (line.empty()) {
            p.id=player;
            d.cards=cards;
            p.deck=d;
            g.players.push_back(p);
            cards.clear();
            i=0;
            continue;
        }
        if (i==0) {
            player=stoi(line.substr(7,line.find(':')-7));
        } else {
            ll value = stoi(line);
            cards.push_back(value);
        }
        i++;
    }
    p.id=player;
    d.cards=cards;
    p.deck=d;
    g.players.push_back(p);

    // part 2
    g.winner = playGame(g,{});
    cout<<score(g)<<endl;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);
    cout<<duration.count()<<endl;
    return 0;
}
```

## Day 23

### Part 2: 

```cpp
struct Node {
    Node* next;
    ll val;
};

struct Cups {
    Node* cups;
    vector<Node*> indexVec;

    explicit Cups() {
        vector<Node*> indices(1000001, new Node);
        indexVec = indices;
    }
};

struct Game {
    Cups c;
    ll destVal;
    Node* rmNode;
    Node* destNode;
    set<ll> picked;
};

Game g;

void playGame(bool part2) {
    Node* next;
    Node* head;
    head = g.c.cups;
    if (part2) {
        for (int i = 0;i<8;i++) {
            g.c.cups=g.c.cups->next;
        }
        for (int i = 10;i<=1000000;i++) {
            next = new Node;
            next->val = i;
            g.c.cups->next=next;
            g.c.cups = g.c.cups->next;
            g.c.indexVec[next->val] = next;
        }
        g.c.cups->next = head;
    }
    ll moves = part2 ? 10000000 : 100;
    Node* curCup = head;
    Node* curNode;
    Node* lastNode;
    Node* afterDestNode;
    for (int j = 0;j<moves;j++) {
        // Removing my three nodes
        curNode = curCup->next;
        g.rmNode = curNode;
        for (int i = 0;i<3;i++) {
            g.picked.insert(curNode->val);
            lastNode = curNode;
            curNode = curNode->next;
        }
        if (part2) {
            g.destVal = curCup->val-1==0 ? 1000000 : curCup->val-1;
        } else {
            g.destVal = curCup->val-1==0 ? 9: curCup->val-1;
        }
        curCup->next = curNode;
        lastNode->next = nullptr;
        // Searching for the destination value
        while (g.picked.count(g.destVal)>0) {
            g.destVal--;
            if (g.destVal==0) {
                g.destVal= part2 ? 1000000 : 9;
            }
        }
        // Insert the three nodes after the destination node. 
        g.destNode = g.c.indexVec[g.destVal];
        afterDestNode = g.destNode->next;
        g.destNode->next = g.rmNode;
        while (g.rmNode->next!=nullptr) {
            g.rmNode = g.rmNode->next;
        }
        g.rmNode->next = afterDestNode;
        g.picked.clear();
        curCup = curCup->next;
        Node* test = curCup;
    }
    if (part2) {
        ll res = 1;
        g.destNode = g.c.indexVec[1];
        for (int i = 0;i<2;i++) {
            g.destNode = g.destNode->next;
            cout<<g.destNode->val<<endl;
            res*=g.destNode->val;
        }
        cout<<res<<endl;
    } else {
        string res = "";
        g.destNode = g.c.indexVec[1];
        for (int i = 0;i<8;i++) {
            g.destNode = g.destNode->next;
            res+=to_string(g.destNode->val);
        }
        cout<<res<<endl;
    }
}

int main() {
    auto start = high_resolution_clock::now();
    freopen("inputs/big.txt","r",stdin);
    string input;
    Cups c = Cups();
    getline(cin,input);
    Node* node = new Node;
    Node* next;
    Node* head = node;
    c.cups = node;
    int i = 0;
    for (char ch : input) {
        i++;
        next = new Node;
        node->val=ch-'0';
        // Want the next to be nullptr for the last one. 
        node->next=next;
        c.indexVec[node->val]=node;
        if (i==input.size()) {
            continue;
        }
        node= node->next;
    }
    node->next=head;
    c.cups=head;
    g.c=c;
    // part 1
    cout<<"Part 1: ";
    playGame(false);
    // part 2
    cout<<"Part 2: ";
    playGame(true);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);
    cout<<"Time: "<<duration.count()<<endl;
    return 0;
}
```

## Day 24

### Part 2: 

```cpp
struct v2 {
    int x, y;
    v2() : x(0), y(0) {};
    v2(int x, int y) : x(x), y(y) {};

    v2& operator += (const v2 &b) {
        x+=b.x;
        y+=b.y;
        return *this;
    }

    v2& operator -= (const v2 &b) {
        x-=b.x;
        y-=b.y;
        return *this;
    }

    int& operator[] (int index) {
        if (index==0) {
            return x;
        }
        return y;
    }
};
bool operator == (const v2 &a, const v2 &b) {
    return a.x==b.x && a.y==b.y;
}
bool operator > (const v2 &a, const v2 &b) {
    return a.x!=b.x ? a.x>b.x : a.y>b.y;
}
bool operator < (const v2 &a, const v2 &b) {
    return b>a;
}
bool operator >= (const v2 &a, const v2 &b) {
    return a>b || a==b;
}
bool operator <= (const v2 &a, const v2 &b) {
    return b>=a;
}
bool operator != (const v2 &a, const v2 &b) {
    return !(a==b);
}
v2 operator + (const v2 &a, const v2 &b) {
    return v2(a.x+b.x,a.y+b.y);
}
v2 operator - (const v2 &a, const v2 &b) {
    return v2(a.x-b.x,a.y-b.y);
}

struct Tiles {
    set<v2> tiles;
};

map<string, v2> DIRS = {{"nw", {-1,1}},{"w",{-2,0}},{"sw",{-1,-1}},{"se",{1,-1}},{"e",{2,0}},{"ne",{1,1}}};

struct Directions {
    vector<v2> dirs;

    explicit Directions(const string& line) {
        int i = 0;
        while (i<line.size()) {
            if (line[i]=='s' || line[i]=='n') {
                dirs.push_back(DIRS[line.substr(i,2)]);
                i+=2;
            } else {
                string s(1,line[i]);
                dirs.push_back(DIRS[s]);
                i++;
            }
        }
    }
};

Tiles t;

//////////
//Part 1//
//////////
void flip(vector<v2> directions) {
    v2 p = v2();
    for (v2 pt : directions) {
        p+=pt;
    }
    if (t.tiles.count(p)>0) {
        t.tiles.erase(p);
    } else {
        t.tiles.insert(p);
    }
}

//////////
//Part 2//
//////////
ll hexAutomata(Tiles t) {
    
    for (int i = 0;i<100;i++) {
        Tiles t2;
        set<v2>::iterator it;
        set<v2>::iterator wh;
        for (it=t.tiles.begin();it!=t.tiles.end();it++) {
            map<string,v2>::iterator it2;
            set<v2> whites;
            for (it2=DIRS.begin();it2!=DIRS.end();it2++) {
                v2 delta = it2->second;
                v2 nei = *(it)+delta;
                if (t.tiles.count(nei)==0) {
                    whites.insert(nei);
                }
            }
            ll cntWhite = whites.size();
            ll cntBlack = 6-cntWhite;
            if (cntBlack>0 && cntBlack<3) {
                t2.tiles.insert(*it);
            }
            for (wh=begin(whites);wh!=end(whites);wh++) {
                cntBlack=0;
                for (it2=begin(DIRS);it2!=end(DIRS);it2++) {
                    v2 delta = it2->second;
                    v2 nei = *(wh)+delta;
                    if (t.tiles.count(nei)>0) {
                        cntBlack++;   
                    }
                }
                if (cntBlack==2) {
                    t2.tiles.insert(*wh);
                }
            }
        }
        t=t2;
    }
    return t.tiles.size();
}

int main() {
    freopen("inputs/big.txt","r",stdin);
    string line;
    while (getline(cin,line)) {
        Directions d = Directions(line);
        flip(d.dirs);
    }
    int count = 0;

    set<v2>::iterator it;
    for (it=t.tiles.begin(); it != t.tiles.end();it++) {
        count++;
    }
    // part 1
    cout<<"Part 1: "<<count<<endl;
    // part 2
    cout<<"Part 2: "<<hexAutomata(t)<<endl;
    return 0;
}
```

## Day 25

### Part 2: 

```cpp
struct Keys {
    vector<ll> keys;
    vector<ll> loopSizes;
};

Keys k;
ll cons = 20201227;

ll findLoopSize(ll pubKey) {
    ll num = 1;
    ll loopCnt=0;
    ll subjectNum = 7;
    while (num!=pubKey) {
        num*=subjectNum;
        num%=cons;
        loopCnt++;
    }
    return loopCnt;
}

ll findEncryptionKey() {
    ll encKey = 1;
    ll subjectNum = k.keys[0];
    for (int i = 0;i<k.loopSizes[1];i++) {
        encKey*=subjectNum;
        encKey%=cons;
    }
    return encKey;
}

ll solve() {
    for (int i = 0;i<2;i++) {
        k.loopSizes.push_back(findLoopSize(k.keys[i]));
    }
    return findEncryptionKey();
}

int main() {
    freopen("inputs/big.txt","r",stdin);
    string line;
    while (getline(cin,line)) {
        ll val = stoll(line);
        k.keys.push_back(val);
    }
    cout<<"Part 1:"<<solve()<<endl;
    return 0;
}
```

