# Number Theory

## Useful facts

These facts seem to be useful for enough problems that they are worth memorizing.

If you take from 1 to 1,000,000 here are some useful facts:
- The maximum number of distinct prime factors a number can have is 7.
- The maximum number of total prime factors a number can have is 19.
- The maximum number of divisors a number can have is 240, and it is number 720,720.

for $10^{14}$
- The maximum number of distinct prime factors a number can have is 12.
- The maximum number of total prime factors a number can have is 46.

## Finding all divisors of a number in square root time complexity

```cpp
vector<int> div;
for (int i = 1; i * i <= x; i++) {
    if (x % i == 0) {
        div.emplace_back(i);
        if (i * i < x) div.emplace_back(x / i);
    }
}
```

## Euler Totient Theorem

The euler totient function calculates the number of integers that are coprime with totient(i) from [1, i]

precompute the euler totient function (phi) value from 1 to n in O(nlog(log(n))) time using the sieve of Eratosthenes algorithm.  You can also easily calculate the prefix sum, and also get the prime factorization of all the integers in this.

```cpp
int totient[MAXN], totient_sum[MAXN];
vector<int> primes[MAXN];
void sieve(int n) {
    iota(totient, totient + n, 0LL);
    memset(totient_sum, 0, sizeof(totient_sum));
    for (int i = 2; i < n; i++) {
        if (totient[i] == i) { // i is prime integer
            for (int j = i; j < n; j += i) {
                totient[j] -= totient[j] / i;
                primes[j].emplace_back(i);
            }
        }
    }
    for (int i = 2; i < n; i++) {
        totient_sum[i] = totient_sum[i - 1] + totient[i];
    }
}
```

Solving for phi value for specific n in sqrt(n) time

```cpp
int phi(int n) {
    int result = n;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0)
                n /= i;
            result -= result / i;
        }
    }
    if (n > 1)
        result -= result / n;
    return result;
}
```

## Chinese Remainder Theorem

This theorem is useful for solving systems of congruences. The theorem states that if the moduli are pairwise coprime, then there is a unique solution modulo the product of the moduli.

given moduli n = [m1, m2, m3, m4], and remainders a = [r1, r2, r3, r4] it will find the unique solution modulo m1 * m2 * m3 * m4.

```cpp
// perform the Extended Euclidean Algorithm
// It returns gcd(a, b) and also updates x and y to satisfy ax + by = gcd(a, b)
int extended_gcd(int a, int b, int &x, int &y) {
    if (a == 0) {
        x = 0;
        y = 1;
        return b;
    }
    int x1, y1;
    int gcd = extended_gcd(b % a, a, x1, y1);
    x = y1 - (b / a) * x1;
    y = x1;
    return gcd;
}

// Function to find the modular inverse of a under modulo m using the extended Euclidean algorithm
int mod_inverse(int a, int m) {
    int x, y;
    int gcd = extended_gcd(a, m, x, y);
    if (gcd != 1) {
        throw invalid_argument("Modular inverse does not exist");
    }
    return (x % m + m) % m;
}

// Function to solve the system of congruences using the Chinese Remainder Theorem
// vector n is moduli, and vector a is remainders
int chinese_remainder_theorem(const vector<int>& n, const vector<int>& a) {
    int prod = 1;
    for (int ni : n) {
        prod *= ni;
    }

    int result = 0;
    for (size_t i = 0; i < n.size(); ++i) {
        int ni = n[i];
        int ai = a[i];
        int p = prod / ni;
        int inv = mod_inverse(p, ni);
        result += ai * inv * p;
    }

    return result % prod;
}
```


## Solving a power tower

Using Euler Totient Theorem and Chinese Remainder Theorem to solve power tower, where the power is very large and recursive, and you want to find it modulo some number.

```cpp

```

## Solving an infinite power tower of x

Using Euler Totient Theorem and Chinese Remainder Theorem to solve an infinite power tower of x, where x is a number and you want to find it modulo some number.

Here is an example, if x = 10, you can solve this using the following code.  You have to find the prime factors of 10, which are 2 and 5.  Then you can solve it.

So whatever x is you find the prime factors and have it run chinese remainder theorem whenever you are taking a moduli n that is divisible by one of it's prime factors.  You split all the prime factors that divide it into one moduli and the rest in another. And recursively solve the other one.

It requires the other code from here and exponentiation, change out the 10 for whatever x is.

This could be written to be useable for any x, but for now it was used just for 10.

```cpp
int inf_power(int n) {
    if (n == 1) return 0;
    if (n % 5 == 0 || n % 2 == 0) { // split up and apply CRT
        int m = 1;
        while (n % 5 == 0) {
            m *= 5;
            n /= 5;
        }
        while (n % 2 == 0) {
            m *= 2;
            n /= 2;
        }
        int s1 = 0, s2 = exponentiation(10, calc(phi[n]), n);
        int crt = chinese_remainder_theorem({m, n}, {s1, s2});
        return crt;
    }
    // apply Euler Totient theorem
    int res = exponentiation(10, calc(phi[n]), n);
    return res;
}
```

## Bezout's Identity

It can apply to more than 2 integers, and it can be used to find the greatest common divisor of multiple numbers. The identity states that for any integers a, b, and c, there exist integers x and y such that ax + by = c, where c is the greatest common divisor of a and b.

so ax+by=gcd(a,b), but really multiples of gcd(a,b) as well.
