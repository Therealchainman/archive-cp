def main():
    with open('input.txt', 'r') as f:
        data = f.read()
        stack = []
        for ch in data:
            if stack and stack[-1].lower() == ch.lower() and stack[-1] != ch:
                stack.pop()
            else:
                stack.append(ch)
        return len(stack)
if __name__ == "__main__":
    print(main())