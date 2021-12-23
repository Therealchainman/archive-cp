import collections
import itertools
import math
import re

goal = {
    'A': 2,
    'B': 4,
    'C': 6,
    'D': 8,
}
goalSpaces = set(goal.values())
moveCosts = {
    'A': 1,
    'B': 10,
    'C': 100,
    'D': 1000,
}


def canReach(board, pos, dest):
    a = min(pos, dest)
    b = max(pos, dest)
    for i in range(a, b+1):
        if i == pos:
            continue
        if i in goalSpaces:
            continue
        if board[i] != '.':
            # print(' ', i, board[i][0], 'cannot reach')
            return False
    return True


def roomOnlyContainsGoal(board, piece, dest):
    inRoom = board[dest]
    return len(inRoom) == inRoom.count('.') + inRoom.count(piece) 


def getPieceFromRoom(room):
    for c in room:
        if c != '.':
            return c


def possibleMoves(board, pos):
    piece = board[pos]
    # print(board, pos, piece)
    if pos not in goalSpaces:
        if canReach(board, pos, goal[piece]) and roomOnlyContainsGoal(board, piece, goal[piece]):
            return [goal[piece]]
        return []

    movingLetter = getPieceFromRoom(piece)
    if pos == goal[movingLetter] and roomOnlyContainsGoal(board, movingLetter, pos):
        return []

    possible = []
    for dest in range(len(board)):
        if dest == pos:
            continue
        if dest in goalSpaces and goal[movingLetter] != dest:
            continue
        if goal[movingLetter] == dest:
            if not roomOnlyContainsGoal(board, movingLetter, dest):
                continue
        if canReach(board, pos, dest):
            possible.append(dest)
    return possible


def addToRoom(letter, room):
    room = list(room)
    dist = room.count('.')
    assert dist != 0
    room[dist-1] = letter
    return ''.join(room), dist


def move(board, pos, dest):
    new_board = board[:]
    dist = 0
    movingLetter = getPieceFromRoom(board[pos])
    if len(board[pos]) == 1:
        new_board[pos] = '.'
    else:
        new_room = ''
        found = False
        for c in board[pos]:
            if c == '.':
                dist += 1
                new_room += c
            elif not found:
                new_room += '.'
                dist += 1
                found = True
            else:
                new_room += c
        new_board[pos] = new_room
    
    dist += abs(pos - dest)

    if len(board[dest]) == 1:
        new_board[dest] = movingLetter
        return new_board, dist * moveCosts[movingLetter]
    else:
        new_board[dest], addl_dist = addToRoom(movingLetter, board[dest])
        dist += addl_dist
        return new_board, dist * moveCosts[movingLetter]


def solve(board):
    states = {tuple(board): 0}
    queue = [board]
    while queue:
        # print(len(queue))
        board = queue.pop()
        for pos, piece in enumerate(board):
            if getPieceFromRoom(piece) is None:
                continue
            dests = possibleMoves(board, pos)
            # print('{} ({}) can move to {}'.format(piece, pos, dests))
            for dest in dests:
                new_board, addl_cost = move(board, pos, dest)
                new_cost = states[tuple(board)] + addl_cost
                new_board_tuple = tuple(new_board)
                cost = states.get(new_board_tuple, 9999999)
                if new_cost < cost:
                    # print(board, '->', new_board, ':', new_cost)
                    states[new_board_tuple] = new_cost
                    queue.append(new_board)

    return states

board = ['.', '.', 'AB', '.', 'DC', '.', 'BA', '.', 'DC', '.', '.']
states = solve(board)  # 12240
print(states[('.', '.', 'AA', '.', 'BB', '.', 'CC', '.', 'DD', '.', '.')])

board = ['.', '.', 'ADDB', '.', 'DCBC', '.', 'BBAA', '.', 'DACC', '.', '.']
states = solve(board)  # 44618
print(states[('.', '.', 'AAAA', '.', 'BBBB', '.', 'CCCC', '.', 'DDDD', '.', '.')])