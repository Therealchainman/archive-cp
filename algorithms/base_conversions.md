# base conversions

## Converting from base 10 to base b algorithm

suppose V is in base 10 representation and we want to convert it to base b representation.  Where we have some characters that can be used to represent b representation and solving for the coefficients gives the characters, if you index the characters from 0 to b - 1.  The coefficients are the indexes of the characters that are used to represent the number in base b.

$V = c_{n} * b^{n} + ... + c_{2} * b^{2} + c_{1} * b^{1} + c_{0} * b^{0}$
Do this while V > 0
1. $c_{i} = V \% b$
2. $V = \lfloor \frac{V}{b} \rfloor$

you solve for the coefficients and then you want to reverse the order. 

## Converting from base b to base 10

$V = c_{n} * b^{n} + ... + c_{2} * b^{2} + c_{1} * b^{1} + c_{0} * b^{0}$
and res = 0
do this while V > 0
1. $res = res * b$
2. $res = res + c_{i}$

both of these algorithms will have at most $\log_{b}{V}$ iterations. 
