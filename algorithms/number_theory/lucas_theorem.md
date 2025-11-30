# Lucas Theorem

Lucas theorem allows us to know when the remainder is equal to 0 under modulo 2 for a binomial coefficient. 

But by Lucas’s theorem (or the simple fact that $\binom{m}{k}$  is odd iff every 1-bit of k is also a 1-bit of m).  Or in otherwords k is a submask/subset of the bit representation of m.

From Lucas Theorem we get the following result:

$$\binom{n}{k} \bmod 2 = \begin{cases} 1 & \text{if} \; k \subseteq n \\ 0 & \text{otherwise} \end{cases}$$