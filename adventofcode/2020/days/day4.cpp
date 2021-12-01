#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <sstream>
#include <iostream>

using namespace std;
typedef vector<int> vec;
typedef long long int lli;

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