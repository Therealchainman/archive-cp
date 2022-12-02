from collections import *
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        lose, draw, win = 'X', 'Y', 'Z'
        rock, paper, scissors = 'A', 'B', 'C'
        lose_points, draw_points, win_points = 0, 3, 6
        result_points = {lose: lose_points, draw: draw_points, win: win_points}
        bonus = {rock: 1, paper: 2, scissors: 3}
        play_strat = {rock: {lose: scissors, draw: rock, win: paper}, paper: {lose: rock, draw: paper, win: scissors}, scissors: {lose: paper, draw: scissors, win: rock}}
        score = sum([result_points[strat] + bonus[play_strat[opp][strat]] for opp, strat in map(lambda play: play.split(), data)])
        return score
if __name__ == "__main__":
    print(main())