#include <bits/stdc++.h>
using namespace std;
/*
We are using red, blue and yellow to paint this picture. using vector datastructres to solve this problem to store the strokes, when 
a color is active, and just incrementing when you go from inactive to active for a paint, that will be one stroke. 
*/
int main() {
    int T, N;
    string P;
    cin>>T;
    for (int t=1;t<=T;t++) {
        cin>>N>>P;
        int numStrokes = 0;
        vector<int> B(N+1,0), Y(N+1,0), R(N+1,0);
        for (int i=0;i<N;i++) {
            if (P[i] == 'B' || P[i] ==  'G' || P[i] == 'P' || P[i] == 'A') {
                B[i+1] = 1;
            } 
            if (P[i] == 'Y' || P[i] == 'O' || P[i] == 'G' || P[i] == 'A') {
                Y[i+1] = 1;
            } 
            if (P[i] == 'R' || P[i] == 'O' || P[i] == 'P' || P[i] == 'A') {
                R[i+1] = 1;
            }
        }
        for (int i=1;i<=N;i++) {
            if (B[i] && !B[i-1]) {
                numStrokes++;
            }
            if (Y[i] && !Y[i-1]) {
                numStrokes++;
            }
            if (R[i] && !R[i-1]) {
                numStrokes++;
            }
        }
        printf("Case #%d: %d\n", t, numStrokes);
    }
}