#include <bits/stdc++.h>
using namespace std;
const int MOD = 1e9+7;
/*
inline is useful for functions that are small, but the function call is overhead you don't want, 
that is the function call is greater than the actual execution of function code. 
I actually removed my inline code so this looks like a dumb statement but it was originally for a min 
function. 

So only O and X actually matter, anytime you see one of these characters and they are different than the previous
that means you have to update the delta which is the amount of change for the sum at each index going forward. 
So it is storing the amount of swaps requires for each substring prior. 

Really I saw the solution to this by writing out an example and then I saw how it increased a factor that was like a delta or change
that was then used for each string considered prior
That is you consider all these strings S[0:i], and like there are n substrings in here, but I already know the number of swaps required
by 
*/

int main() {
    int T;
    // freopen("inputs/weak_typing_chapter_2_validation_input.txt", "r", stdin);
    // freopen("outputs/weak_typing_chapter_2_validation_output.txt", "w", stdout);
    freopen("inputs/weak_typing_chapter_2_input.txt", "r", stdin);
    freopen("outputs/weak_typing_chapter_2_output.txt", "w", stdout);
    cin>>T;
    for (int t = 1;t<=T;t++) {
        int N;
        string S;
        cin>>N>>S;
        int sum = 0, last = -1, delta = 0; 
        char prev = 'F';
        for (int i = 0;i<N;i++) {
            if (S[i]=='F' || S[i]==prev) {
                if (S[i]!='F') {
                    last=i;
                }
            } else {
                delta = (delta+last+1)%MOD;
                last = i;
            }
            sum = (sum + delta)%MOD;
            prev = S[i]=='F' ? prev : S[i];
        }
        printf("Case #%d: %d\n", t, sum);
    }
}