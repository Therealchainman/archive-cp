# Sprague Grundy Game Theory

What games does this theorem apply to?
1. two players who move alternatively
1. game consists of states, and possible moves in state do not depend on whose turn it is
1. game ends when a player cannot make a move
1. the game is finite
1. the players have complete information about the states and allowed moves, and there is no randomness in the game


calculate a Grundy number



mex is minimum excludant


To calculate the Grundy value of a given state you need to:

Get all possible transitions from this state

Each transition can lead to a sum of independent games (one game in the degenerate case). Calculate the Grundy value for each independent game and xor-sum them. Of course xor does nothing if there is just one game.

After we calculated Grundy values for each transition we find the state's value as the  
$\text{mex}$  of these numbers.

If the value is zero, then the current state is losing, otherwise it is winning.

In comparison to the previous section, we take into account the fact that there can be transitions to combined games. We consider them a Nim with pile sizes equal to the independent games' Grundy values. We can xor-sum them just like usual Nim according to Bouton's theorem.


```py

```