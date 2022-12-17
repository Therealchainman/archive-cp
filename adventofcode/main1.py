from collections import defaultdict

def main():
    with open("input.txt", 'r') as f:
        data = f.read()
        rocks = [[['#','#','#','#']], [['.','#','.'],['#','#','#'],['.','#','.']], [['.','.','#'],['.','.','#'],['#','#','#']], [['#'],['#'],['#'],['#']], [['#','#'],['#','#']]]
        width = 8
        index = 0
        n = len(data)
        total = 2022
        height = 0
        done = set()
        for k in range(total):
            rock = rocks[k%len(rocks)]
            coords = []
            R, C = len(rock), len(rock[0])
            for r in range(R):
                for c in range(C):
                    if rock[~r][c] == '#':
                        coords.append((r + height + 4, c + 3))
            while True:
                blow = data[index]
                if blow == '>':
                    if not any(c == width-1 or (r, c+1) in done for r, c in coords): 
                        for i in range(len(coords)):
                            r, c = coords[i]
                            coords[i] = (r, c+1) # shift right by 1
                else:
                    if not any(c == 1 or (r, c-1) in done for r, c in coords): 
                        for i in range(len(coords)):
                            r, c = coords[i]
                            coords[i] = (r, c-1) # shift left by 1
                index = (index + 1) % n
                if any(r == 1 or (r-1, c) in done for r, c in coords): break # stopped reach ground or another rock
                for i in range(len(coords)):
                    r, c = coords[i]
                    coords[i] = (r-1, c) # shift down by 1

            height = max([height] + [r for r, c in coords])
            done.update(coords)
        return height
        


if __name__ == '__main__':
    print(main())
"""
2022 rocks have stopped falling

7 units wide

so okay I have 5 rocks that are going to be repeating, 
so all I need to know is basically the shape of the rocks

rock appears so that it left edge is two nits away from left wall
and its bottome edge is three units above the highest rock
"""