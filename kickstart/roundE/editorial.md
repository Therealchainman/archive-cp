



## Increasing Sequence Card Game


### Solution: brute force by simulating number of swaps with N! permutations O(N*N!)

```c++
int main() {
    int T, N;
    cin>>T;
    for (int t = 1;t<=T;t++) {
        cin>>N;
        vector<int> orig(N);
        iota(orig.begin(),orig.end(),0);
        int gamesPlayed = 0, swaps = 0;
        do {
            for (int i =0, cur=-1;i<N;i++) {
                swaps += (orig[i]>cur);
                cur = max(cur, orig[i]);
            }
            gamesPlayed++;
        } while(next_permutation(orig.begin(),orig.end()));
        double E = (double)swaps/gamesPlayed;
        printf("Case #%d: %f\n",t,E);
    }
}
```

### Solution: harmonic series O(N)
The tricky part is coming up with the harmonic series there are a couple methods
(1) Observation from brute force solution, you can see it is adding 1/2+1/3+1/4... 
(2) linearity of expectation so we have the expectation value of 5 swaps, we just add all the expectation values.

```c++
int main() {
    int T, N;
    cin>>T;
    for (int t = 1;t<=T;t++) {
        cin>>N;
        double E = 0;
        for (int i = 1;i<=N;i++) {
            E += (1.0/i);
        }
        printf("Case #%d: %f\n",t,E);
    }
}
```

### Solution: Harmonic series approximation in O(logN)

```c++
int main() {
    long long T, N;
    cin>>T;
    for (int t = 1;t<=T;t++) {
        cin>>N;
        double E = 0;
        if (N<1000) {
            for (int i = 1;i<=N;i++) {
                E += 1.0/i;
            }
        } else {
            const double gamma = 0.5772156649;
            E += gamma + log(N)+ 1.0/(2*N) + 1.0/(12*N*N);
        }
        printf("Case #%d: %f\n",t,E);
    }
}
```