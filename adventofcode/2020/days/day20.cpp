#include "../libraries/aoc.h"

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