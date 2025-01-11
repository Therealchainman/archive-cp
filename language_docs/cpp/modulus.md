# Modulus

In C++ dealing with modulus is a bit different from python.  This is the way to handle negative numbers and return value in range [0, m).]

```cpp
int modulus(int x, int m) {
    return (x % m + m) % m;
}
```