#include <bits/stdc++.h>
using namespace std;
/*
All you need is to put a 1 in for each intersection or element in the matrix. 
And then you can add the remaining value for A an B.  

I mean seems like that is all that would create a valid one

The only time it is not valid is when the minimal number of intersections necessary to cross
to get to follow the paths is when the value req        int left = 0, right = 0; // initialize the start with left and right hand at 0uired is smaller than if you add up 1 for the
shortest path you can take in the town. 
*/

const string possible = "Possible";
const string impossible = "Impossible";
int main() {
    int T;
    // freopen("inputs/traffic_control_validation_input.txt", "r", stdin);
    // freopen("outputs/traffic_control_validation_output.txt", "w", stdout);
    freopen("inputs/traffic_control_input.txt", "r", stdin);
    freopen("outputs/traffic_control_output.txt", "w", stdout);
    cin>>T;
    for (int t = 1;t<=T;t++) {
        int N, M, A, B;
        cin>>N>>M>>A>>B;
        int minimalIntersections = N+M-1;
        if (A<minimalIntersections || B<minimalIntersections) {
            printf("Case #%d: %s\n", t, impossible.c_str());
            continue;
        }
        vector<vector<int>> grid(N, vector<int>(M, 1));
        grid[N-1][M-1]=A-minimalIntersections+1;
        grid[N-1][0] = B-minimalIntersections+1;
        printf("Case #%d: %s\n", t, possible.c_str());
        for (int i = 0;i<N;i++) {
            for (int j = 0;j<M;j++) {
                printf("%d ", grid[i][j]);
            }
            printf("\n");
        }
    }
}

