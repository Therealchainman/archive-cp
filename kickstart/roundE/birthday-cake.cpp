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

        (r1,c1)
                        (r2,c2)
        */  
        // names are for if they are leftMostTop, a cut that comes from the top of the rectangle and is to the left 
        // of the delicious rectangle, or the left edge of the delicious rectangle. 
        int leftMostTop = (r2+K-1)/K, leftMostBot = (R-r1+K)/K, topMostLeft = (c2+K-1)/K, topMostRight = (C-c1+K)/K;
        int rightMostTop = leftMostTop, rightMostBot = leftMostBot, botMostLeft = topMostLeft, botMostRight = topMostRight;
        int width = c2-c1+1, height = r2-r1+1;
        // Try all 8 variants for cutting out the partial delicious rectangle
        if (r2==R) {
            botMostLeft = 0, botMostRight = 0;
        }
        if (r1==1) {
            topMostLeft = 0, topMostRight = 0;
        }
        if (c2==C) {
            rightMostTop = 0, rightMostBot = 0;
        }
        if (c1==1) {
            leftMostTop = 0, leftMostBot = 0;
        }
        if (r2!=R) {
            leftMostBot+=((width+K-1)/K);
            topMostLeft+=((width+K-1)/K);
            topMostRight+=((width+K-1)/K);
            leftMostTop+=((width+K-1)/K);
            rightMostBot+=((width+K-1)/K);
            rightMostTop+=((width+K-1)/K);
        } 
        if (r1!=1) {
            leftMostBot+=((width+K-1)/K);
            leftMostTop+=((width+K-1)/K);
            rightMostBot+=((width+K-1)/K);
            botMostLeft+=((width+K-1)/K);
            botMostRight+=((width+K-1)/K);
            rightMostTop+=((width+K-1)/K);
        }
        if (c2!=C) {
            leftMostBot+=((height+K-1)/K);
            topMostLeft+=((height+K-1)/K);
            topMostRight+=((height+K-1)/K);
            leftMostTop+=((height+K-1)/K);
            botMostLeft+=((height+K-1)/K);
            botMostRight+=((height+K-1)/K);
        }
        if (c1!=1) {
            topMostLeft+=((height+K-1)/K);
            topMostRight+=((height+K-1)/K);
            rightMostBot+=((height+K-1)/K);
            botMostLeft+=((height+K-1)/K);
            botMostRight+=((height+K-1)/K);
            rightMostTop+=((height+K-1)/K);
        }
        int numCuts = min({topMostLeft, topMostRight, botMostLeft, botMostRight, leftMostBot, leftMostTop, rightMostBot, rightMostTop}); // to reach the fully delicious rectangle
        // start with verticals then compute horizontals
        int vertHor = (width-1)*((height+K-1)/K) + (height-1)*width;
        // start with horizontals and then compute verticals
        int horVert = (height-1)*((width+K-1)/K) + (width-1)*height;
        int cnt = min(vertHor, horVert);
        printf("Case #%d: %d\n", t, numCuts+cnt);
    }
}
/*
1
6 6 2
3 2
4 4

*/