'''
Create the most difficult string of length n, where X and O oscillate
'''
def create(n, keys):
    res = ""
    for i in range(0,n,len(keys)):
        res += keys
    assert(len(res)==n)
    return res


if __name__ == '__main__':
    with open('outputs/FOXstring.txt', 'w') as f:
        f.write(str(800000))
        f.write("\n")
        f.write(create(800000, "OX"))