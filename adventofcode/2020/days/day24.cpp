#include "../libraries/aoc.h"

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