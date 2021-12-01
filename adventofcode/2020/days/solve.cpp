#include "../libraries/aoc.h"

// // This checks that the matrices are equivalent to each other. 
// bool isNotChanged(vector<vector<char>>& M1, vector<vector<char>>& M2) {
//     int R = M1.size(), C = M2.size();
//     int count = 0;
//     int totalCells = 0;
//     for (int r = 0;r<R;r++) {
//         for (int c = 0;c<C;c++) {
//             totalCells++;
//             if (M1[r][c] == M2[r][c]) {
//                 count++;
//             }
//         }
//     }
//     return count==totalCells;
// }

// void searchSeats(vector<vector<char>>& M) {
//     int R = M.size(), C= M[0].size();
//     for (int r = 0;r<R;r++) {
//         for (int c = 0;c<C;c++) {
//             if (M[r][c]=='.') {
//                 continue;
//             }
//             int cntOccupied = 0;
//             for (pair<int,int> pa : dirs) {
                
//             }
//         }
//     }
// }


// int main() { 
//     vector<pair<ll,ll>> dirs = {{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,-1},{1,-1},{-1,1}};
//     freopen("inputDay11.txt","r",stdin);
//     string x;
//     vector<vector<char>> M;
//     ll r = 0;
//     while (getline(cin,x)) {        
//         M.push_back(vector<char>{});
//         for (int i = 0;i<x.size();i++) {
//             M[r].push_back(x[i]);
//         }
//         r++;
//     }
//     ll R = M.size(), C = M[0].size();
//     vector<vector<char>> MM = M;
//     for (int i = 0;i<100;i++) {
//         for (int r = 0;r<R;r++) {
//             for (int c = 0;c<C;c++) {
//                 if (M[r][c]=='.') {
//                     continue;
//                 }
//                 ll cntOccupied = 0;
//                 ll cntFloor = 0;
//                 for (pair<ll,ll> pa : dirs) {
//                     ll dx= pa.first;
//                     ll dy = pa.second;
//                     ll x = r;
//                     ll y = c;
//                     bool occupied = false;
//                     while (x>=0 && x<R && y>=0 && y<C && M[x][y]=='.') {
//                         if (M[x][y]=='#') {
//                             occupied = true;
//                         }
//                         x+=dx;
//                         y+=dy;
//                     }
//                     if (occupied) {
//                         cntOccupied++;
//                     }
//                 }   
//                 if (M[r][c]=='L' && cntOccupied==0) {
//                     MM[r][c] = '#';
//                 } else if (M[r][c]=='#' && cntOccupied>=5) {
//                     MM[r][c] = 'L';
//                 }
//             }
//         }
//         if (isNotChanged(M, MM)) {
//             break;
//         }
//         M = MM;
//     }
    
//     ll res = 0;
//     for (int r = 0;r<R;r++) {
//         for (int c = 0;c<C;c++) {
//             if (M[r][c]=='#') {
//                 res++;
//             }
//         }
//     }
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

    map<int, set<rule>> rulesByCols;
    bool valid;
    for (int c = 0;c<p.myTicket.size();c++) {
        for (rule r : p.rules) {
            valid = true;
            for (vector<int> curTicket : p.nearByTickets) {
                bool ticketValid = r.valueIsValid(curTicket[c]);
                valid &=ticketValid;
                if (!valid) {
                    break;
                }
            }
            if (valid) {
                rulesByCols[c].insert(r);
            }
        }
    }
    /*
    This code is intended to basically intended to find the columns that have so on and so on.  This is 
    very challenging though.  
    */
    set<rule> usedRules;
    while (true) {
        // focus on understanding this code.  
        map<int, set<rule>>::iterator it = find_if(rulesByCols.begin(),rulesByCols.end(),
        [&usedRules] (pair<int,set<rule>> elem) {
            return elem.second.size()==1;
        });
        if (it==rulesByCols.end()) {
            break;
        }
        rule curRule = *it->second.begin();
        usedRules.insert(curRule);
        for_each(rulesByCols.begin(),rulesByCols.end(), 
        [curRule] (pair<const int,set<rule>> rs) {
            if (rs.second.size()>1) {
                rs.second.erase(curRule);
            }
        });
    }

    ll res = 1;
    for (int i = 0;i<p.myTicket.size();i++) {
        if (rulesByCols[i].size()==1 && rulesByCols[i].begin()->isDeparture) {
            res*=p.myTicket[i];
        }
    }
    return res;
}


int main() {

    freopen("big.txt","r",stdin);
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