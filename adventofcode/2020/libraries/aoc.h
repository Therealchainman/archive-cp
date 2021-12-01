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
#define rep(i,n) for (i = 0; i < n; ++i) 
#define REP(i,k,n) for (i = k; i <= n; ++i) 
#define REPR(i,k,n) for (i = k; i >= n; --i)
#define pb push_back
#define all(a) a.begin(), a.end()
#define fastio ios::sync_with_stdio(0); cin.tie(0); cout.tie(0)
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

ll mod (ll a, ll b) {
    return (a % b + b) %b;
}
////////////
// Day 16//
///////////

////////////
// Day 17//
///////////
struct p3 {
    int x, y, z;
    p3() : x(0), y(0), z(0) {}
    p3(int x, int y, int z) : x(x), y(y), z(z) {}

    p3& operator+= (const p3 &b) {
        x+b.x;
        y+b.y;
        z+b.z;
        return *this;
    }

    p3& operator-= (const p3 &b) {
        x-b.x;
        y-b.y;
        z-b.z;
        return *this;
    }

    int& operator[] (int index) {
        if (index == 0) {
            return x;
        } 
        if (index == 1) {
            return y;
        }
        return z;
    }
    int operator[] (int index) const {
        if (index == 0) {
            return x;
        } 
        if (index == 1) {
            return y;
        }
        return z;
    }
};

bool operator == (const p3 &a, const p3 &b) {
    return a.x==b.x && a.y==b.y && a.z==b.z;
}
bool operator > (const p3 &a, const p3 &b) {
    return a.x!=b.x ? a.x>b.x : a.y!=b.y ? a.y>b.y : a.z<b.z;
}
bool operator < (const p3 &a, const p3 &b) {
    return b>a;
}
bool operator != (const p3 &a, const p3 &b) {
    return !(a==b);
}
bool operator >= (const p3 &a, const p3 &b) {
    return !(a<b);
}
bool operator <= (const p3 &a, const p3 &b) {
    return !(a>b);
}
p3 operator + (const p3 &a, const p3 &b) {
    return p3(a.x+b.x,a.y+b.y,a.z+b.z);
}
p3 operator - (const p3 &a, const p3 &b) {
    return p3(a.x-b.x,a.y-b.y,a.z-b.z);
}

struct p4 {
    int x, y, z,w;
    p4() : x(0), y(0), z(0), w(0) {}
    p4(int x, int y, int z, int w) : x(x), y(y), z(z), w(w) {}

    p4& operator+= (const p4 &b) {
        x+b.x;
        y+b.y;
        z+b.z;
        w+b.w;
        return *this;
    }

    p4& operator-= (const p4 &b) {
        x-b.x;
        y-b.y;
        z-b.z;
        w-b.w;
        return *this;
    }

    int& operator[] (int index) {
        if (index == 0) {
            return x;
        } 
        if (index == 1) {
            return y;
        } 
        if (index == 2) {
            return z;
        }
        return w;
    }
    int operator[] (int index) const {
        if (index == 0) {
            return x;
        } 
        if (index == 1) {
            return y;
        }
        if (index == 2) {
            return z;
        }
        return w;
    }
};

bool operator == (const p4 &a, const p4 &b) {
    return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w;
}
bool operator > (const p4 &a, const p4 &b) {
    return a.x!=b.x ? a.x>b.x : a.y!=b.y ? a.y>b.y : a.z!=b.z ? a.z>b.z : a.w>b.w;
}
bool operator < (const p4 &a, const p4 &b) {
    return b>a;
}
bool operator != (const p4 &a, const p4 &b) {
    return !(a==b);
}
bool operator >= (const p4 &a, const p4 &b) {
    return !(a<b);
}
bool operator <= (const p4 &a, const p4 &b) {
    return !(a>b);
}
p4 operator + (const p4 &a, const p4 &b) {
    return p4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);
}
p4 operator - (const p4 &a, const p4 &b) {
    return p4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w);
}

// Using for day 18
ll evaluate(ll x, ll y, char oper) {
    return oper == '+' ? x + y : x*y;
}

struct number {
    vector<string> lines;
};

////////////
// Day 22//
///////////
struct Deck {
   deque<ll> cards;
};

bool operator == (const Deck &a, const Deck &b) {
    return a.cards==b.cards;
}

struct Player {
    int id=0;
    Deck deck;
};

bool operator == (const Player &a, const Player &b)  {
    return a.id==b.id && a.deck==b.deck;
}
bool operator < (const Player &a, const Player &b) {
    return a.id < b.id;
}
bool operator > (const Player &a, const Player &b) {
    return !(a<b);
}
bool operator != (const Player &a, const Player &b) {
    return !(a==b);
}
// struct Game {
//     vector<Player> players;
//     Player winner;
// };