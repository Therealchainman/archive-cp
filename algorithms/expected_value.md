# EXPECTED VALUE

Informally, the expected value is the arithmetic mean of a large number of independently selected outcomes of a random variable.

## EXPECTED MAXIMUM EXAMPLE PROBLEM

Twilight Sparkle was playing Ludo with her friends Rainbow Dash, Apple Jack and Flutter Shy. But she kept losing. Having returned to the castle, Twilight Sparkle became interested in the dice that were used in the game.

The dice has m faces: the first face of the dice contains a dot, the second one contains two dots, and so on, the m-th face contains m dots. Twilight Sparkle is sure that when the dice is tossed, each face appears with probability . Also she knows that each toss is independent from others. Help her to calculate the expected maximum number of dots she could get after tossing the dice n times.

$E(x) = $

```py
m, n = map(int, input().split())
prob = lambda x: pow((x / m), n)
res = sum(i * (prob(i) - prob(i - 1)) for i in range(1, m + 1))
print(res)  
```