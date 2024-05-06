# XOR Hashing

This is a technique to hash integers using xor of all the integers.  And allows you to distinguish every subset that is different.  That is if the hash1 = hash2 that means they must be the same set of integers.  Allows you to uniquely identify sets of integers.

In order to prevent hash collision, you have to give each integer a random 64 bit integer.

This one definitely works for when you have less than 1,000 integers, although I don't actually know the limit, I imagine it actually can go very large. 

```cpp
mt19937_64 rnd(12341234);

// this is how you assign the random integer to each type of value from 0 to K - 1.
for (int i = 0; i < K; i++) {
    vals[i] = rnd();
}
```