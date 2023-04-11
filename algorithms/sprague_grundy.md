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


What is the value for a state that is a winning state and losing state?  I think it is 0 for losing state and 1 for winning state.

anything greater than 0 is a winning state, if you are forced to use a 0 than you are in losing state. 

look for pattern in these problems since that is only way it can be solved. 

nim state is the nim sum s = x1^x2^x3^...^xn
s = 0 is losing state
s > 0 winning state

for grundy numbers there is a state graph representation of the game. 
In addition you calculate the mex({0,1,3}) = 2

when do you use a state graph to represent it? 

```py
def main(N, L, R, piles):
    total = sum(piles)
    def grundy(idx, remaining, num_piles):
        # winning state for player
        if num_piles == 1 and L <= remaining <= R: return 1 
        # losing state for player
        if num_piles == 1 and remaining < L: return 0
        grundy_numbers = set()
        for i in range(N):
            for take in range(L, min(R, piles[i]) + 1):
                piles[i] -= take
                new_num_piles = num_piles - (1 if piles[i] < L else 0)
                grundy_numbers.add(grundy(idx + 1, remaining - take, new_num_piles))
                piles[i] += take
        res = next(dropwhile(lambda i: i in grundy_numbers, range(100_000)))
        return res
    num_piles = sum((1 for p in piles if p >= L))
    grundy_number = grundy(0, total, num_piles)
    return "First" if grundy_number > 0 else "Second"
```