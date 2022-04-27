BINARY = '10101010'
def main():
    while True:
        print(BINARY, flush=True)
        N = int(input())
        if N == 0:
            return True
        elif N == -1:
            return False

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        if not main():
            break