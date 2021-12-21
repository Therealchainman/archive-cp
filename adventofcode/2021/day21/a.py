class DiracDice:
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            data = f.read().split("\n")
            return int(data[0].split(': ')[1])-1, int(data[1].split(': ')[1])-1
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
if __name__ == '__main__':
    dd = DiracDice()
    print(dd.run())
