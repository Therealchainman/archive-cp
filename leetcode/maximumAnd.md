

First solution is just general bit manipulation, deriving the known And and then computing current And values for those that can be set 
and contribute to and of array.  

```c++
int freq[32];
class Solution{
    public:
    int maxAnd(int N, vector<int> A){
        memset(freq,0,sizeof(freq));
        for (int &x : A) {
            for (int i = 0;i<32;i++) {
                if ((x>>i)&1) {
                    freq[i]++;
                }
            }   
        }
        int knownAnd = 0;
        for (int i = 0;i<32;i++) {
            if (freq[i]==N) {
                knownAnd += (1<<i);
            }
        }
        int resultAnd = knownAnd;
        for (int &x : A) {
            int curAnd = 0;
            for (int i = 0;i<32;i++) {
                if (!((x>>i)&1) && freq[i]==N-1) {
                    curAnd += (1<<i);
                }
            }
            resultAnd = max(resultAnd, knownAnd + curAnd);
        }
        return resultAnd;
    }
};
```