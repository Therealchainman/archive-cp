"""
Creates a solution template
"""

import sys

problem_counts = {"atcoder": 7, "codeforces": 6, "leetcode": 4}

def create(contest, name, number, division = None):
    if contest == "atcoder":
        if name == "beginner":
            path = f"{contest}/abc/{name}{number}.md"
            sys.stdout = open(path, 'w')
            print(f"# Atcoder Beginner Contest {number}")
        elif name == "regular":
            path = f"{contest}/arc/{name}{number}.md"
            sys.stdout = open(path, 'w')
            print(f"# Atcoder Regular Contest {number}")
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
        if name == "weekly":
            print(f"# Leetcode Weekly Contest {number}")
        else:
            print(f"# Leetcode BiWeekly Contest {number}")
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
    # contest, name, number, div = "atcoder", "beginner", 343, None
    # contest, name, number, div = "atcoder", "regular", 173, None
    # contest, name, number, div = "leetcode", "biweekly", 125, None
    # contest, name, number, div = "leetcode", "weekly", 388, None
    # contest, name, number, div = "codeforces", "round", 933, 3
    # contest, name, number, div = "codeforces", "educational", 162, 2
    create(contest, name, number, division = div)