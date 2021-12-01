#include "../libraries/aoc.h"

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