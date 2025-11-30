# Series

## Harmonic Series

1 + 1/2 + 1/3 + 1/4 + 1/5 + ... 1 / N = log(N)

Often times the harmonic series comes up in algorithms and complexity analysis.  If you are cutting the data down by these factors N / 1 + N / 2 + ... you get O(NlogN) time complexity which is very useful for solving some problems.  It can be easily mistaken for O(N^2), but just look for that crucial fact that you are dividing the size of the input data by larger number each time.  It also feels a little counter intuitive to accept the above as being log(N) iterations. 

## Infinite Geometric Series

This is a very famous one that sums to 2, where the ratio is 1/2.  

1 + 1/2 + 1/4 + 1/8 + 1/16 + ... = 2

s = a / (1 - r)
a = 1, r = 1/2

Often times it will be N + N / 2 + N / 4 +... = 2N, where you just multiple by N.