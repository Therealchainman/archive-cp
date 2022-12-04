
def main():
    with open('test_output.txt', 'r') as f1, open('output.txt', 'r') as f2:
        for i, (line1, line2) in enumerate(zip(f1, f2), start = 1):
            line1 = line1.split()
            line2 = line2.split()
            if line1 != line2:
                print(f'line {i}: WA')
                print(len(line1), len(line2))
                for j, (c1, c2) in enumerate(zip(line1, line2), start = 1):
                    if c1 != c2:
                        print(f'char {j}: actual: {c1} != mine: {c2}')

if __name__ == '__main__':
    main()