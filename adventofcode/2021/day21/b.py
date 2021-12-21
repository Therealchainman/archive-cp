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
    def run(self):
        initial_pos1, initial_pos2 = map(int, self.data_loader())
        return max(self.simulateUniverses(0, 0, initial_pos1, initial_pos2))
if __name__ == '__main__':
    dd = DiracDice()
    print(dd.run())

