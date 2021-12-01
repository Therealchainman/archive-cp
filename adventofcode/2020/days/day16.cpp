#include "../libraries/aoc.h"

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