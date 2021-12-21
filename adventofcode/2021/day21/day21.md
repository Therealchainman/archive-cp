Dice game problem

# Part 1

Solution: Simulate the game via iteration

```py
def run(self):
    pos1, pos2= map(int, self.data_loader())
    dice, score1, score2, turn, cnt = -1, 0, 0, 0, 0
    while score1<1000 and score2<1000:
        cnt+=1
        if turn==0:
            moves = 0
            for _ in range(3):
                dice = (dice+1)%100
                moves += dice +1
            pos1 = (pos1+moves)%10
            score1 += pos1+1
        else:
            moves = 0
            for _ in range(3):
                dice = (dice+1)%100
                moves += dice +1
            pos2 = (pos2+moves)%10
            score2 += pos2+1     
        turn^=1
    return min(score1,score2)*cnt*3
```

# Part 2

Solution: Iterative DP with states

```py
class DiracDice:
    def __init__(self):
        self.states = [[[[[0 for _ in range(2)] for _ in range(10)] for _ in range(10)] for _ in range(21)] for _ in range(21)]
        # states[score1][score2][p1][p2][turn] = the number of universes with this state
        self.wins1, self.wins2 = 0, 0
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            data = f.read().split("\n")
            return int(data[0].split(': ')[1])-1, int(data[1].split(': ')[1])-1
    def run(self):
        initial_pos1, initial_pos2 = self.data_loader()
        print(initial_pos1,initial_pos2)
        self.states[0][0][initial_pos1][initial_pos2][0] = 1
        curStates = set([(0,0,initial_pos1,initial_pos2,0)])
        while curStates:
            newStates = set()
            for score1, score2, pos1, pos2, turn in curStates:
                cnt = self.states[score1][score2][pos1][pos2][turn]
                self.states[score1][score2][pos1][pos2][turn] = 0
                if turn==0:
                    for i in range(1,4):
                        for j in range(1,4):
                            for k in range(1,4):
                                pos = (pos1+i+j+k)%10
                                score = score1+pos+1
                                if score>=21:
                                    self.wins1+=cnt
                                else:
                                    self.states[score][score2][pos][pos2][turn^1] += cnt
                                    newStates.add((score,score2,pos,pos2,turn^1))
                else:
                    for i in range(1,4):
                        for j in range(1,4):
                            for k in range(1,4):
                                pos = (pos2+i+j+k)%10
                                score = score2+pos+1
                                if score>=21:
                                    self.wins2+=cnt
                                else:
                                    self.states[score1][score][pos1][pos][turn^1] += cnt
                                    newStates.add((score1,score,pos1,pos,turn^1))
            curStates = newStates
        return max(self.wins1,self.wins2)
if __name__ == '__main__':
    dd = DiracDice()
    print(dd.run())
```

Solution: Recursive DP with states and memoization with cache


```py
from functools import cache
class DiracDice:
    def __init__(self):
        self.states = [[[[[0 for _ in range(2)] for _ in range(10)] for _ in range(10)] for _ in range(21)] for _ in range(21)]
        # states[score1][score2][p1][p2][turn] = the number of universes with this state
        self.wins1, self.wins2 = 0, 0
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            data = f.read().split("\n")
            return int(data[0].split(': ')[1])-1, int(data[1].split(': ')[1])-1
    @cache
    def simulateUniverses(self, score1, score2, pos1, pos2, turn):
        wins1, wins2 = 0, 0
        if turn==0:
            for i in range(1,4):
                for j in range(1,4):
                    for k in range(1,4):
                        pos = (pos1+i+j+k)%10
                        score = score1+pos+1
                        if score>=21:
                            wins1 +=1
                        else:
                            nwins1, nwins2 =  self.simulateUniverses(score, score2, pos, pos2, turn^1)
                            wins1 += nwins1
                            wins2 += nwins2
        else:
            for i in range(1,4):
                for j in range(1,4):
                    for k in range(1,4):
                        pos = (pos2+i+j+k)%10
                        score = score2+pos+1
                        if score>=21:
                            wins2 +=1
                        else:
                            nwins1, nwins2 =  self.simulateUniverses(score1, score, pos1, pos, turn^1)
                            wins1 += nwins1
                            wins2 += nwins2
        return wins1, wins2
    def run(self):
        initial_pos1, initial_pos2 = map(int, self.data_loader())
        return max(self.simulateUniverses(0, 0, initial_pos1, initial_pos2, 0))
if __name__ == '__main__':
    dd = DiracDice()
    print(dd.run())
```


Solution: Recursive DP but reduced the arguments

Use the fact that you switch players each turn, so you are computing the other wins and other losses, and the opponent losses count towards your wins.
And you switch who will be the current player in the next recursive call.  This way I don't need boolean for storing the turn.  

```py
@cache
def simulateUniverses(self, cur_score, other_score, cur_pos, other_pos):
    if cur_score>=21 or other_score>=21:
        return cur_score>=21, other_score>=21
    wins, losses = 0, 0
    for i in range(1,4):
        for j in range(1,4):
            for k in range(1,4):
                pos = (cur_pos+i+j+k)%10
                score = cur_score +pos+1
                other_wins,other_losses = self.simulateUniverses(other_score, score, other_pos, pos)
                wins += other_losses
                losses += other_wins
    return wins, losses
```