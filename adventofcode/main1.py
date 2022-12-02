def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        rock, paper, scissors = 'X', 'Y', 'Z'
        rock_opp, paper_opp, scissors_opp = 'A', 'B', 'C'
        lose_points, draw_points, win_points = 0, 3, 6
        bonus = {rock: 1, paper: 2, scissors: 3}
        points = {rock_opp: {rock: draw_points, paper: win_points, scissors: lose_points}, paper_opp: {rock: lose_points, paper: draw_points, scissors: win_points}, 
        scissors_opp: {rock: win_points, paper: lose_points, scissors: draw_points}}
        score = sum([bonus[you] + points[opp][you] for opp, you in map(lambda play: play.split(), data)])
        return score
if __name__ == "__main__":
    print(main())