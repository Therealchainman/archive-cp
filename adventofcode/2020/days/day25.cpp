#include "../libraries/aoc.h"

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