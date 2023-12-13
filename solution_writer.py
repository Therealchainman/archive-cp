"""
Creates a solution template
"""

import sys

problem_counts = {"atcoder": 7, "codeforces": 6, "leetcode": 4}

def create(contest, name, number, division = None):
    if contest == "atcoder":
        if name == "beginner":
            path = f"{contest}/abc/{name}{number}.md"
            print(f"# Atcoder Beginner Contest {number}")
        elif name == "regular":
            path = f"{contest}/arc/{name}{number}.md"
            print(f"# Atcoder Regular Contest {number}")
        sys.stdout = open(path, 'w')
        print()
        for _ in range(problem_counts[contest]):
            print("## ")
            print()
            print("### Solution 1: ")
            print()
            print("```py")
            print()
            print("```")
            print()
    elif contest == "codeforces":
        path = f"{contest}/div{division}/{name}_{number}.md"
        sys.stdout = open(path, 'w')
        print(f"# Codeforces {name.capitalize()} {number} Div {division}")
        print()
        for _ in range(problem_counts[contest]):
            print("## ")
            print()
            print("### Solution 1: ")
            print()
            print("```py")
            print()
            print("```")
            print()
    elif contest == "leetcode":
        path = f"{contest}/{name}/{name}{number}.md"
        sys.stdout = open(path, 'w')
        print(f"# Leetcode Weekly Contest {number}")
        print()
        for _ in range(problem_counts[contest]):
            print("## ")
            print()
            print("### Solution 1: ")
            print()
            print("```py")
            print()
            print("```")
            print()
    sys.stdout.close()

if __name__ == '__main__':
    contest, name, number, div = "atcoder", "beginner", 332, None
    # contest, name, number, div = "atcoder", "regular", 169, None
    # contest, name, number, div = "leetcode", "biweekly", 119, None
    # contest, name, number, div = "leetcode", "weekly", 375, None
    # contest, name, number, div = "codeforces", "round", 900, 3
    # contest, name, number, div = "codeforces", "educational", 155, 2
    create(contest, name, number, division = div)