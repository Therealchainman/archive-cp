#include "../libraries/aoc.h"

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