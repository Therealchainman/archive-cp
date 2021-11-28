#include <bits/stdc++.h>
using namespace std;
/*
hashmap problem to allow distribution among the overlapping students that select both day i and day j. Use this 
to try to bring some students from the overlap if not enough students.  
*/

int main() {
    int T, N, a;
    cin>>T;
    while (T--) {
        cin>>N;
        unordered_map<int,int> numStudentsEachDay; // day -> count of students
        vector<vector<int>> students(N, vector<int>(5,0)); // student -> days
        unordered_map<int, unordered_map<int,int>> overlaps; // day1 -> (day2 -> count of students)
        for (int i = 0;i<N;i++) {
            for (int j = 0;j<5;j++) {
                cin>>a;
                students[i][j]=a;
                numStudentsEachDay[j] += a;
            }
            for (int k = 0;k<4;k++) {
                for (int j = k+1;j<5;j++) {
                    if (students[i][k]&students[i][j]) {
                        overlaps[k][j]++;
                        overlaps[j][k]++;
                    }
                }
            }
        }
        bool found = false;
        for (int i = 0;i<4;i++) {
            for (int j = i+1;j<5;j++) {
                int num1 = numStudentsEachDay[i]-overlaps[i][j], num2 = numStudentsEachDay[j]-overlaps[i][j], used = 0;
                if (num1<N/2) {
                    int diff = min(overlaps[i][j], N/2-num1);
                    num1 += diff;
                    used += diff;
                }
                if (num1<N/2) {continue;}
                if (num2<N/2) {
                    int diff = min(overlaps[i][j]-used, N/2-num2);
                    num2 += diff;
                    used += diff;
                }
                if (num2<N/2) {continue;}
                found = true;
                break;
            }
        }
        if (found) {
            cout<<"YES"<<endl;
        } else {
            cout<<"NO"<<endl;
        }
    }
}