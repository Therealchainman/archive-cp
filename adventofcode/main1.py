class Delta:
    def __init__(self, delta: str):
        self.val = int(delta[1:]) if '+' in delta else -int(delta[1:])
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        return sum([Delta(val).val for val in data])
if __name__ == "__main__":
    print(main())