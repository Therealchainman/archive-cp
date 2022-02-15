import random
import copy
from collections import Counter
def simulateGame(cards):
    hand = []
    hand.append(cards.pop())
    while len(cards)>0:
        card = cards.pop()
        if card>hand[-1]:
            hand.append(card)
    return len(hand)

def numCards(x, numGames):
    cards = list(range(x))
    totalHands = 0
    counts = Counter()
    for _ in range(numGames):
        random.shuffle(cards)
        hands = simulateGame(copy.deepcopy(cards))
        counts[hands]+=1
        totalHands += hands
    print(sorted([(x, y/numGames) for x, y in counts.items()]))
    return totalHands / numGames

if __name__ == '__main__':
    for i in range(10,101,10):
        print(i, numCards(i, 10000))
'''
10 cards
[1:91, 2:293, 3:322, 4:209, 5:71, 6:13, 7:1]
[(1, 0.113), (2, 0.286), (3, 0.305), (4, 0.196), (5, 0.075), (6, 0.023), (7, 0.002)]
2.919
20 cards
[1:41, 2:193, 3:266, 4:256, 5:148, 6:70, 7:20, 8:6]
[(1, 0.044), (2, 0.183), (3, 0.269), (4, 0.263), (5, 0.153), (6, 0.063), (7, 0.02), (8, 0.003), (9, 0.002)]
3.597
30 cards
[1:27, 2:131, 3:239, 4:268, 5:182, 6:87, 7:45, 8:16, 9:3, 10:2]
[(1, 0.034), (2, 0.145), (3, 0.236), (4, 0.245), (5, 0.187), (6, 0.092), (7, 0.04), (8, 0.017), (9, 0.004)]
4.0
'''