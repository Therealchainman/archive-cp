# Fast Fourier Transform algorithm (FFT)

A fast Fourier transform (FFT) is an algorithm that computes the discrete Fourier transform (DFT) of a sequence, or its inverse (IDFT).

## Convolution of two polynomials

Note this is over a finite field, that is it is modulo some prime number p

This program performs polynomial multiplication using Number Theoretic Transform (NTT) to efficiently count certain properties of a given sequence. Let's analyze the components in detail.

This namespace implements the Number Theoretic Transform (NTT), a modular Fast Fourier Transform (FFT).
P = 998244353 is a prime modulus commonly used in competitive programming.
R = 3 is a primitive root of P (used for NTT).
IR = (P + 1) / 3 is the modular inverse of R, used for inverse NTT.

`initNTT`
Finds the smallest power of 2 (N = 2^L) greater than or equal to l (length of the polynomial).
Computes IN = 1/N % P using modular inverse, required for inverse NTT.
Precomputes rev array for bit-reversal permutation used in iterative NTT.

`NTT`
Iterative butterfly operation:
Processes pairs of elements (x, y) using roots of unity.
Uses modular arithmetic to ensure values stay within P.
Applies radix-2 Cooley-Tukey FFT algorithm.

Given n numbers, the code:
Creates a frequency array a where a[x] = 1 if x appears.
Computes polynomial multiplication A(x) * A(x) using NTT.
Extracts valid (x, y) pairs such that x + y is present.
Computes the final count of valid pairs.

Uses NTT (modular FFT) for fast polynomial multiplication.
Efficient O(N log N) approach instead of naive O(N²).
Uses modular arithmetic with P = 998244353 for performance.
Solves pair counting problems efficiently.

This example is being used to count pairs of ordered triplets such that A + C = 2 * B, but think of it as A + C = k

```cpp
namespace NTT{
    const int P = 998244353, R = 3, IR = (P + 1) / 3;

    int L, N, IN;
    vector<int> rev;

    int qpow(int b, int k) {
        int ret = 1;
        while(k > 0) {
            if(k & 1) ret = static_cast<int64>(ret) * b % P;
            b = static_cast<int64>(b) * b % P;
            k >>= 1;
        }
        return ret;
    }

    void initNTT(int l) {
        L = 0;
        while((1 << L) < l) L ++;
        N = 1 << L;
        IN = qpow(N, P - 2);
        
        rev = vector<int>(N);

        for(int i = 0; i < N; i ++)
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (L - 1));
    }

    void NTT(vector<int> &a, bool type) {
        a.resize(N);
        for(int i = 0; i < N; i ++)
            if(i < rev[i])
                swap(a[i], a[rev[i]]);
        
        for(int i = 1; i < N; i *= 2) {
            int64 g = qpow(type ? R : IR, (P - 1) / (i * 2));
            for(int j = 0; j < N; j += i * 2) {
                int64 gn = 1;
                for(int k = 0; k < i; k ++, gn = gn * g % P) {
                    int x = a[j + k], y = a[i + j + k] * gn % P;
                    a[j + k] = (x + y) % P;
                    a[i + j + k] = (x - y + P) % P;
                }
            }
        }
    }

    vector<int> convolution(vector<int> a, vector<int> b) {
        int len = (int)a.size() + (int)b.size() - 1;
        NTT(a, 1);
        NTT(b, 1);
        for(int i = 0; i < N; i ++)
            a[i] = a[i] * static_cast<int64>(b[i]) % P;
        NTT(a, 0);
        a.resize(len);
        for(auto &x : a)
            x = static_cast<int64>(x) * IN % P;
        return a;
    }
}

vector<int> polynomialMultiplication(const vector<int>& a, const vector<int>& b) {
    NTT::initNTT(a.size() + b.size() - 1);
    return NTT::convolution(a, b);
}

const int MAXN = 1e6 + 5;
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(MAXN, 0);
    for (int i = 0, x; i < N; ++i) {
        cin >> x;
        A[x] = 1;
    }
    int64 ans = 0;
    vector<int> B = polynomialMultiplication(A, A);
    for (int i = 0; i < B.size(); i += 2) {
        if (A[i / 2]) ans += (B[i] - 1) / 2;
    }
    cout << ans << endl;
}
```

## pythonic atcoder library implementation

This is an implementation that allows convolution using the fast fourier tranform algorithm.  It can be used for convolutions (such as multiplication of polynomials)

This one requires a modulo m to be passed through.  Now I noticed it doesn't work with a lot of m values.  It does happen to work for the m = 998244353 thought, which is a common one.

```py
from itertools import product

class FFT:
    """
    https://github.com/shakayami/ACL-for-python/blob/master/convolution.py
    """
    def primitive_root_constexpr(self, m):
        if m == 2: return 1
        if m == 167772161: return 3
        if m == 469762049: return 3
        if m == 754974721: return 11
        if m == 998244353: return 3
        divs = [0] * 20
        divs[0] = 2
        x = (m - 1) // 2
        while x % 2 == 0: x //= 2
        i = 3
        cnt = 1
        while i * i <= x:
            if x % i == 0:
                divs[cnt] = i
                cnt += 1 
                while x % i == 0: x //= i
            i += 2
        if x > 1:
            divs[cnt] = x
            cnt += 1
        g = 2
        while 1:
            ok = True
            for i in range(cnt):
                if pow(g, (m - 1) // divs[i], m) == 1: 
                    ok = False
                    break
            if ok: return g
            g += 1
    # bit scan forward, finds the rightmost set bit? maybe? 
    def bsf(self, x):
        res = 0
        while x % 2 == 0:
            res += 1
            x //= 2
        return res
    rank2 = 0
    root = []
    iroot = []
    rate2 = []
    irate2 = []
    rate3 = []
    irate3 = []
    def __init__(self, MOD):
        self.mod = MOD
        self.g = self.primitive_root_constexpr(self.mod)
        self.rank2 = self.bsf(self.mod - 1)
        self.root = [0] * (self.rank2 + 1)
        self.iroot = [0] * (self.rank2 + 1)
        self.rate2 = [0] * self.rank2
        self.irate2 = [0] * self.rank2
        self.rate3 = [0] * (self.rank2 - 1)
        self.irate3 = [0] * (self.rank2 - 1)
        self.root[self.rank2] = pow(self.g, (self.mod - 1) >> self.rank2, self.mod)
        self.iroot[self.rank2] = pow(self.root[self.rank2], self.mod - 2, self.mod)
        for i in range(self.rank2 - 1, -1, -1):
            self.root[i] = (self.root[i + 1] ** 2) % self.mod
            self.iroot[i] = (self.iroot[i + 1] ** 2) % self.mod
        prod = iprod = 1
        for i in range(self.rank2 - 1):
            self.rate2[i] = (self.root[i + 2] * prod) % self.mod
            self.irate2[i] = (self.iroot[i + 2] * iprod) % self.mod
            prod = (prod * self.iroot[i + 2]) % self.mod
            iprod = (iprod * self.root[i + 2]) % self.mod
        prod = iprod = 1
        for i in range(self.rank2 - 2):
            self.rate3[i] = (self.root[i + 3] * prod) % self.mod
            self.irate3[i] = (self.iroot[i + 3] * iprod) % self.mod
            prod = (prod * self.iroot[i + 3]) % self.mod
            iprod = (iprod * self.root[i + 3]) % self.mod
    def butterfly(self, a):
        n = len(a)
        h = (n - 1).bit_length()
        LEN = 0
        while LEN < h:
            if h - LEN == 1:
                p = 1 << (h - LEN - 1)
                rot = 1
                for s in range(1 << LEN):
                    offset = s << (h - LEN)
                    for i in range(p):
                        l = a[i + offset]
                        r = a[i + offset + p] * rot
                        a[i + offset] = (l + r) % self.mod
                        a[i + offset + p] = (l - r) % self.mod
                    rot *= self.rate2[(~s & -~s).bit_length() - 1]
                    rot %= self.mod
                LEN += 1
            else:
                p = 1 << (h - LEN - 2)
                rot = 1
                imag = self.root[2]
                for s in range(1 << LEN):
                    rot2 = (rot * rot) % self.mod
                    rot3 = (rot2 * rot) % self.mod 
                    offset = s << (h - LEN)
                    for i in range(p):
                        a0 = a[i + offset]
                        a1 = a[i + offset + p] * rot
                        a2 = a[i + offset + 2 * p] * rot2
                        a3 = a[i + offset + 3 * p] * rot3
                        a1na3imag = (a1 - a3) % self.mod * imag
                        a[i + offset] = (a0 + a1 + a2 + a3) % self.mod
                        a[i + offset + p] = (a0 + a2 - a1 - a3) % self.mod
                        a[i + offset + 2 * p] = (a0 - a2 + a1na3imag) % self.mod
                        a[i + offset + 3 * p] = (a0 - a2 - a1na3imag) % self.mod
                    rot *= self.rate3[(~s & -~s).bit_length() - 1]
                    rot %= self.mod
                LEN += 2
    def butterfly_inv(self, a):
        n = len(a)
        h = (n - 1).bit_length()
        LEN = h
        while LEN:
            if LEN == 1:
                p = 1 << (h - LEN)
                irot = 1
                for s in range(1 << (LEN - 1)):
                    offset = s << (h - LEN + 1)
                    for i in range(p):
                        l = a[i + offset]
                        r = a[i + offset + p]
                        a[i + offset] = (l + r) % self.mod
                        a[i + offset + p] = (l - r) * irot % self.mod
                    irot *= self.irate2[(~s & -~s).bit_length() - 1]
                    irot %= self.mod
                LEN -= 1
            else:
                p = 1 << (h - LEN)
                irot = 1
                iimag = self.iroot[2]
                for s in range(1 << (LEN - 2)):
                    irot2 = (irot * irot) % self.mod
                    irot3 = (irot * irot2) % self.mod
                    offset = s << (h - LEN + 2)
                    for i in range(p):
                        a0 = a[i + offset]
                        a1 = a[i + offset + p]
                        a2 = a[i + offset + 2 * p]
                        a3 = a[i + offset + 3 * p]
                        a2na3iimag = (a2 - a3) * iimag % self.mod
                        a[i + offset] = (a0 + a1 + a2 + a3) % self.mod
                        a[i + offset + p] = (a0 - a1 + a2na3iimag) * irot % self.mod
                        a[i + offset + 2 * p] = (a0 + a1 - a2 - a3) * irot2 % self.mod
                        a[i + offset + 3 * p] = (a0 - a1 - a2na3iimag) * irot3 % self.mod
                    irot *= self.irate3[(~s & -~s).bit_length() - 1]
                    irot %= self.mod
                LEN -= 2
    def convolution(self, a, b):
        n = len(a)
        m = len(b)
        if not (a) or not (b): return []
        if min(n, m) <= 40: # naive solution
            res = [0] * (n + m - 1)
            for i, j in product(range(n), range(m)):
                res[i + j] += a[i] * b[j]
                res[i + j] %= self.mod
            return res
        z = 1 << (n + m - 2).bit_length()
        a = a + [0] * (z - n)
        b = b + [0] * (z - m)
        self.butterfly(a)
        self.butterfly(b)
        c = [(a[i] * b[i]) % self.mod for i in range(z)]
        self.butterfly_inv(c)
        iz = pow(z, self.mod - 2, self.mod)
        for i in range(n + m - 1):
            c[i] = (c[i] * iz) % self.mod
        return c[: n + m - 1]
```

## Implementation in C++ 

This does not contain modulo

It also currently is expecting max size of the arrays to be less than 2^19 or 524,288 roughly

```cpp
typedef complex<double> cd;
const int SIZE = 1<<19;
const double PI = acos(-1);

int n, m;
vector<cd> A(SIZE), B(SIZE);

void fft(vector<cd> &a, bool invert) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}
```

```cpp
fft(A, false);
fft(B, false);
for (int i = 0; i < SIZE; i++) {
    A[i] *= B[i];
}
fft(A, true);
for (int i = 0; i < n + m - 1; i++) {
    cout << llround(A[i].real()) << " ";
}
```

You can modify it to work with different sizes other than SIZE, but it needs to be a power of two.

```cpp
vector<cd> A = vector<cd>(poly1.begin(), poly1.end()), B = vector<cd>(poly2.begin(), poly2.end());
int N = A.size(), M = B.size();
int sz = 1;
while (sz < N + M - 1) sz <<= 1;
A.resize(sz, 0);
B.resize(sz, 0);
fft(A, false);
fft(B, false);
for (int i = 0; i < sz; ++i) {
    A[i] *= B[i];
}
fft(A, true);
vector<int64> ans(N + M - 1);
for (int i = 0; i < N + M - 1; ++i) {
    ans[i] = llround(A[i].real());
}
return ans;
```

Explanation:

### FFT on two sequences
fft(A, false) and fft(B, false).  These lines compute the FFT of two sequences (or polynomials) A and B.  The false parameter means it is going to perform forward FFT.  Transforms both sequences from time domain to frequency domain. 

### point-wise multiplicaton of transformed sequences

The for loop iterates through the transformed sequences A and B, performing element-wise multiplication: A[i] *= B[i];. In the frequency domain, this is equivalent to the convolution of the original time-domain signals, which in turn corresponds to polynomial multiplication.

### Performing Inverse FFT

fft(A, true).  This line computes the inverse FFT of the sequence A. The true parameter indicates that the function is performing the inverse FFT. This step transforms the sequence back from the frequency domain to the time domain. The result is the product of the original polynomials represented by A and B.

