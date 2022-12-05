import string
def main():
    with open('input.txt', 'r') as f:
        data = f.read()
        best = len(data)
        for rem_ch in string.ascii_lowercase:
            stack = []
            for ch in data:
                if ch.lower() == rem_ch:
                    continue
                if stack and stack[-1].lower() == ch.lower() and stack[-1] != ch:
                    stack.pop()
                else:
                    stack.append(ch)
            best = min(best, len(stack))
        return best
if __name__ == "__main__":
    print(main())