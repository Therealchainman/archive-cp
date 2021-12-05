def winningScore(game):
    for i in range(5):
        markedRow = sum(1 for j in range(5) if game[i][j] < 0)
        if markedRow==5:
            return sum(-game[i][j] for j in range(5))
        markedCol = sum(1 for j in range(5) if game[j][i] < 0)
        if markedCol==5:
            return sum(-game[j][i] for j in range(5))
    return -1

if __name__ == '__main__':
    with open("inputs/input.txt", "r") as f:
        calls = list(map(lambda x: int(x.replace('\n', '')), f.readline().split(",")))
        boards = []
        while f.readline():
            boards.append([list(map(int, f.readline().split())) for _ in range(5)])
        avail = set(range(len(boards)))
        def bingoCalls():
            for num in calls:
                for index in set(avail):
                    board = boards[index]
                    for i in range(5):
                        for j in range(5):
                            if board[i][j] == num:
                                board[i][j] = -num
                    score = winningScore(board)
                    if score != -1:
                        avail.remove(index)
                        print(num)
                        if len(avail) == 0:
                            unMarkedScore = sum(board[i][j] for i in range(5) for j in range(5) if board[i][j] > 0)
                            return unMarkedScore*num
        print(bingoCalls())
        
