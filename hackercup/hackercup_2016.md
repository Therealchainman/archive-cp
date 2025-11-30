# Meta Hacker Cup 2016

# Round 2

## 

```py

```

## Problem B: Carnival Coins

Assumption is we are dealing with a discrete random variable

In this case the discrete random variable has exactly two possible values 1 (success) and 0 (failure)

In fact this is a binomial distribution probability (visualize image would be bell curve shaped)

binomial distribution is describing the distribution of binary data from a finite sample.

Then there exists a distribution, and using the probability mass function for binomial distribution.  

The function solves gives the probability of m successes given n trials with probability of success being p.  

The probability of success is constant in this example. 

Suppose you win a prize if at least K of outcomes are successes? 

If you play optimally what is expected number of prizes you'd win. 

expected value of winning a prize,  so this is the expected value of indicator variable that is either you win or lose prize so it is either 1 or 0, expected value of something occurring

prob(i, j) = the probability of i successes in j trials
p(i) = prob(k, k) + prob(k, k + 1) + ... + prob(k, i)
probability of at least k given i trials

E(i) = the maximum expected value for i trials
E(i) = max(E(i), E(j) + P(i - j))
i - j is the number of trials that are being added, so basically you are adding some probability of achieving at least k for some trials to some already maximized expected value.  In that case you maximize for the current expected value with i trials
And then use that for i + 1 trials and i + 2 trials and so on. 

Adding experiments to optimize the expected value or summation of probabilities of at least K. 

Another way to view it is you are spliting the data into some experiments with n1, n2, n3, ... nk trials. and this is the optimial split possibly.  The recurrence relation above would find this optimal split.  But not find the actual splits but just the resulting expected value of the optimal splits, which is just going to be the maximum expected value. 
n1 + n2 + ... + nk = n
E(n) = P(K <= x <= n1) + P(K <= x <= n2) + ... + P(K <= x <= nk) 


some more notes on this problem. I think this maybe switches some things up, but it's describing the same solution. 

probability of getting k successes or heads
n is number of trials
P(X = k) 

standard dp to calculate probability of j heads when i coins flipped
dp[i][j] = probability when i coins flipped with j heads

dp[0][0] = 1
just know you need to set the probability for dp[i][0], that is probability of 0 heads for flipping i coins.  TTTTTT
dp[i][j] = dp[i - 1][j - 1] * p + dp[i - 1][j] * (1 - p)


dp2[i] = maximum expected number of prizes after flipping i coins


given i coins at current step, take a partition step j
where i - j is the number of coins to flip, so say you flip these coins,
you need how many heads
dp2[0] = 0
dp2[i] = dp2[j] + dp[i - j] where 0 <= j <= i

so suppose you have the maximum expected value for flipping j coins, but you have remaining i - j coins to flip, so you want to find the max of those over all possible number of heads right? 

Note be careful when input double type it causes issue, just avoid read() all together it seems to work.

```cpp
const double SMALL = numeric_limits<double>::min();
vector<vector<double>> prob;
vector<double> dp, expected;

void solve() {
    int N, K;
    double p;
    cin >> N >> K >> p;
    prob.assign(N + 1, vector<double>(N + 1, 0.0));
    dp.assign(N + 1, 0.0);
    expected.assign(N + 1, SMALL);
    prob[0][0] = 1.0;
    for (int i = 1; i <= N; i++ ) { // flip i coins
        for (int j = 0; j <= i; j++) { // exactly j heads
            prob[i][j] = (1 - p) * prob[i - 1][j];
            if (j > 0) prob[i][j] += p * prob[i - 1][j - 1];
        }
    }
    // dp[i] calculates the probability when flipping i coins to get at least K heads
    for (int i = 1; i <= N; i++) {
        for (int j = K; j <= i; j++) {
            dp[i] += prob[i][j];
        }
    }
    expected[0] = 0.0;
    // maximum expected value
    for (int i = 1; i <= N; i++) { // up to i coins flippped
        for (int j = 0; j < i; j++ ) { 
            expected[i] = max(expected[i], expected[j] + dp[i - j]);
        }
    }
    printf("%.12f\n", expected.end()[-1]);
}

int32_t main() {
	ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    string in = "inputs/" + name;
    string out = "outputs/" + name;
    freopen(in.c_str(), "r", stdin);
    freopen(out.c_str(), "w", stdout);
    int T;
    cin >> T;
    for (int i = 1; i <= T ; i++) {
        printf("Case #%d", i);
        solve();
    }
    return 0;
}
```

## 

```py

```

##

```py

```

##

```py

```

##

```py

```

##

```py

```