# base conversions

## Converting from base 10 to base b algorithm

suppose V is in base 10 representation and we want to convert it to base b representation.  Where we have some characters that can be used to represent b representation and solving for the coefficients gives the characters, if you index the characters from 0 to b - 1.  The coefficients are the indexes of the characters that are used to represent the number in base b.

$V = c_{n} * b^{n} + ... + c_{2} * b^{2} + c_{1} * b^{1} + c_{0} * b^{0}$
Do this while V > 0
1. $c_{i} = V \% b$
2. $V = \lfloor \frac{V}{b} \rfloor$

you solve for the coefficients and then you want to reverse the order. 

## Example using b = 95 

b = 95 allows you to use every single readable ascii character from [32, 126].  So yeah it is interesting representation.

num_chars = b^num_chars, how many characters in the this base representation. 

Because you are going beyond the 0-9 digits, you explore using string, to use the characters. 

```cpp
string convert_base(int h, int b) {
    string s;
    for (int i = 0; i < num_chars; i++) {
        char c = (h % b) + 32;
        s += c;
        h /= b;
    }
    return s;
}
```

## Converting from base b to base 10

$V = c_{n} * b^{n} + ... + c_{2} * b^{2} + c_{1} * b^{1} + c_{0} * b^{0}$
and res = 0
do this while V > 0
1. $res = res * b$
2. $res = res + c_{i}$

both of these algorithms will have at most $\log_{b}{V}$ iterations.

## Getting the ith digit in a base b

There is an easy way to get the corresponding coefficient under a base.  This is that method. 

c0x^0+c1x^1+c2x^2+c3x^3+c4x^4, I want to get the 2nd term, I need to divide by x^2 to get
c2 + c3x + c4x^2, and then just take modulus x to get c2.

if x = 10, you see how it works so easily.  

```cpp
// p = b^i
// mask is the integer representation
int get_digit(int mask, int p, int b = 3) {
    return (mask / p) % b;
}
```

## Encoding a State in Base-*b* as an Integer

### 1. The Idea of Base-*b* Encoding

Suppose you have a sequence (vector) of length `m`:


You can pack this sequence into a single integer using base-$b$ encoding:

$$
\text{code} = d_0 \cdot b^{m-1} + d_1 \cdot b^{m-2} + \dots + d_{m-2} \cdot b^1 + d_{m-1} \cdot b^0
$$

This is exactly how base-10 numbers are represented — we just use base $b$ instead.

---

### 2. Building the Code Incrementally

Rather than computing all powers of $b$ explicitly, you can **build the integer incrementally** by “shifting and appending” each digit. Start from $\text{code} = 0$, and for each digit $d$:

code = code * b + d

Multiplying by $b$ shifts the current digits to the left (in base $b$).

Adding the new digit $d$ appends it to the least-significant position.

#### Example

Let $b = 5$, $m = 4$, and the digits be $[2, 0, 4, 1]$.

We compute the encoded integer step by step:

| Step         | Operation              | Resulting Code | Base-5 Representation |
|--------------|------------------------|----------------|------------------------|
| Start        | —                      | $0$            | `[]`                   |
| Append `2`   | $0 \cdot 5 + 2$        | $2$            | `[2]`                  |
| Append `0`   | $2 \cdot 5 + 0$        | $10$           | `[2, 0]`               |
| Append `4`   | $10 \cdot 5 + 4$       | $54$           | `[2, 0, 4]`            |
| Append `1`   | $54 \cdot 5 + 1$       | $271$          | `[2, 0, 4, 1]`         |

So the final encoded integer is:

$$
\text{code} = 2 \cdot 5^3 + 0 \cdot 5^2 + 4 \cdot 5^1 + 1 \cdot 5^0 = 271
$$
