"""
Creates a solution template
"""

import sys

problem_counts = {"atcoder": 7, "codeforces": 6, "leetcode": 4}

def create(contest, name, number, division = None):
    if contest == "atcoder":
        path = f"{contest}/{name}{number}.md"
        sys.stdout = open(path, 'w')
        print(f"# Atcoder Beginner Contest {number}")
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
        print(f"# Codeforces {name.capitalize()} {number}")
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
    # contest, name, number, div = "atcoder", "beginner", 320, None
    # contest, name, number, div = "leetcode", "biweekly", 113, None
    # contest, name, number, div = "leetcode", "weekly", 363, None
    # contest, name, number, div = "codeforces", "round", 894, 3
    # contest, name, number, div = "codeforces", "educational", 154, 2
    create(contest, name, number, division = div)