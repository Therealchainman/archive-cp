# Generating Function

G.f. is the projection of combinatorial structures. 

The ordinary generating function (o.g.f.) for a sequence \(a_0, a_1, a_2, \ldots\) is the formal power series
$$A(x) = a_0 + a_1 x + a_2 x^2 + \ldots = \sum_{n=0}^{\infty} a_n x^n.$$

usually you call the x^n the formal powers of the symbol x. 

So a formal power series can be viewed as a generalization of polynomials where the number of terms is allowed to be infinite, and differ from usual power series by the absence of convergence requirements.

You can also think of a formal power series just as a type of algebraic object, where we do not care wheter it converges.  It is just an infinite polynomial with symbol x and coefficients $a_n$.

Formal power series are widely used in combinatorics for representing sequences of integers as generating functions. In this context, a recurrence relation between the elements of a sequence may often be interpreted as a differential equation that the generating function satisfies.

A generating function is usually a formal power series that we interpret as encoding a sequence the coefficients $a_n$ are meaningful data.

A generating function is a formal structure that is closely related to a numerical sequence, but allows us to manipulate the sequence as a single entity. 

Formal Definition of Generating Function:
For a sequence \(a_0, a_1, a_2, \ldots\), the generating function is defined as the formal power series
$$G(x) = \sum_{n=0}^{\infty} a_n x^n.$$
The coefficient \(a_n\) of \(x^n\) in the generating function represents the nth term of the sequence.

To further illustrate the difference between these two concepts is the following: 
Every generating function is a formal power series, but not every formal power series is being. used as a generating function. 

### Exponential Generating Function

The exponential generating function (e.g.f.) for a sequence \(a_0, a_1, a_2, \ldots\) is defined as
$$E(x) = a_0 + a_1 \frac{x}{1!} + a_2 \frac{x^2}{2!} + \ldots = \sum_{n=0}^{\infty} a_n \frac{x^n}{n!}.$$

An important property of an EGF is tha tthe convolution of EGF is a weighted confolution of an EGF.

$$
\left[ \frac{x^k}{k!} \right] \hat{A}(x)\,\hat{B}(x)
= \left[ \frac{x^k}{k!} \right]
\left( \sum_{i=0}^{\infty} a_i \frac{x^i}{i!} \right)
\left( \sum_{j=0}^{\infty} b_j \frac{x^j}{j!} \right)
= k! \sum_{i+j=k} \frac{a_i}{i!}\,\frac{b_j}{j!}
= \sum_{i+j=k} \binom{k}{i}\,a_i b_j .
$$

Use exponential generating function because:
- The combinatorial formula involves factors of $\frac{1}{i_m!}$
- Exponential generating functions naturally encode these factorials. 

The reason you have to is because the positions 1, to N are all distinct(labelled)
So for each value m, choosing which positions get value m is like taking a labeled subset of positions. 

The EGF for "take a set of size i of labeled elements" is $\frac{x^i}{i!}$

How to get even only $$B(x)=\sum_{n \geq 0, \text{n is even}} \frac{x^n}{n!}$$

You can get this by doing the following trick to get just even terms for a formal power series. 

Define: $A(x)=\sum_{n=0}^{\infin} a_n x^n$

It turns out that $$\sum_{\text{even n}} a_n x^n = \frac{A(x) + A(-x)}{2}$$

Given that $A(x) = e^x$, you get the following for $B(x) = \frac{e^x + e^{-x}}{2}$

Similarly for odd you have the following
$$\sum_{\text{n odd}} a_n x^n = \frac{A(x) - A(-x)}{2}$$
And this gives you for $Y(x) = \frac{e^x - e^{-x}}{2}$

Now extracting coefficient you can use the following rule. 

$$e^{ax} = \sum_{n=0}^{\infin} a^n \frac{x^n}{n!}$$

## Bivariate Generating Functions

bivariate generating function is a formal power series in two variables, $

### Strategy

Start with understanding the sequence, then generating function comes from that. 


sometimes the EGFs are ugyl as functions and you have to use them as formal power series, and oftentimes that forces you to perform multiplication with convolutions. 




