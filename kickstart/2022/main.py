def main():
    L, N = map(int, input().split())
    laps = pos = 0
    start = None
    for _ in range(N):
        dist, dir = input().split()
        dist = int(dist)
        remainingLaps = pos if dir == 'A' else (L - pos)%L
        sign = 1 if dir == 'C' else -1
        pos = (pos + sign*dist) % L
        if dist >= remainingLaps:
            currentLaps = 1 if remainingLaps > 0 and start == dir else 0
            dist -= remainingLaps
            currentLaps += dist // L
            laps += currentLaps
            start = dir
    return laps
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')