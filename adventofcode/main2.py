from collections import defaultdict
import sys
sys.stdout = open('output.txt', 'w')
def main():
    with open("input.txt", 'r') as f:
        data = f.read()
        rocks = [[['#','#','#','#']], [['.','#','.'],['#','#','#'],['.','#','.']], [['.','.','#'],['.','.','#'],['#','#','#']], [['#'],['#'],['#'],['#']], [['#','#'],['#','#']]]
        width = 8
        index = 0
        n = len(data)
        # total = 1000000000000
        total = 100000
        height = 0
        heights = [0]
        diffs = []
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
            heights.append(height)
            diffs.append(heights[-1] - heights[-2])
            done.update(coords)
        print(','.join(map(str, diffs)))
        return height
        


if __name__ == '__main__':
    print(main())

sys.stdout.close()

"""

"""