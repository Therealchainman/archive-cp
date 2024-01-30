# Advanced Techniques

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## One Bit Positions

### Solution 1:  DFT, FFT, convolution, polynomial multiplication

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

signed main() {
    string s;
    cin >> s;
    n = s.size();
    for (int i = 0; i < n; i++) {
        A[i] = s[i] - '0';
        B[n - i - 1] = s[i] - '0';
    }
    fft(A, false);
    fft(B, false);
    for (int i = 0; i < SIZE; i++) {
        A[i] *= B[i];
    }
    fft(A, true);
    for (int i = n; i < 2 * n - 1; i++) {
        cout << llround(A[i].real()) << " ";
    }
    cout << endl;
}
```

## Signal Processing

### Solution 1:  DFT, FFT, convolution, polynomial multiplication

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

signed main() {
    cin >> n >> m;
    for (int i = 0; i < n; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < m; i++) {
        cin >> B[m - i - 1];
    }
    fft(A, false);
    fft(B, false);
    for (int i = 0; i < SIZE; i++) {
        A[i] *= B[i];
    }
    fft(A, true);
    for (int i = 0; i < n + m - 1; i++) {
        cout << llround(A[i].real()) << " ";
    }
    cout << endl;
}
```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```

## 

### Solution 1:  

```py

```