#include <bits/stdc++.h>
using namespace std;
/*
This is the solution for when K=1 always
*/
int main() {
    int T, R, C, K, r1,c1,r2,c2;
    cin>>T;
    for (int t=1;t<=T;t++) {
        cin>>R>>C>>K>>r1>>c1>>r2>>c2;;
        /*
        Cut it out the special rectangle from the entire rectangle
        */
        int top = r1-1, bot = R-r2, left = c1-1, right = C-c2;
        int numCuts = min({top,bot,left,right}); // to reach the delicious rectangle
        int width = c2-c1+1, height = r2-r1+1;
        if (r2!=R) {
            numCuts+=width;
        } 
        if (r1!=1) {
            numCuts+=width;
        }
        if (c2!=C) {
            numCuts+=height;
        }
        if (c1!=1) {
            numCuts+=height;
        }
        numCuts += (width-1)*height;
        numCuts += (height-1)*width;
        printf("Case #%d: %d\n", t, numCuts);
    }
}