# Random

## randint in C++

uniform random number generator.
Generate a random integer in some some range [l, r].

```cpp
random_device rd;
mt19937_64 gen(rd());
int randint(int l, int r) {
    uniform_int_distribution<int> dist(l, r);
    return dist(gen);
}
```