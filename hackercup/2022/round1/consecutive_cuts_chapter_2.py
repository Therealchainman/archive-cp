import sys
import argparse
problem = sys.argv[0].split('.')[0]
def main():
    N, K = map(int, f.readline().split())
    text = ''.join(f.readline().split())
    pattern = ''.join(f.readline().split())
    sentinel_char = '$'
    if K == 0: return "YES" if text == pattern else "NO"
    if N == 2: 
        if text[0] == text[1] or pattern[0] == pattern[1]:
            return "YES" if text == pattern else "NO"
        if K%2==0: return "YES" if text == pattern else "NO"
        else: return "YES" if text != pattern else "NO"
    text += text
    # IF K IS EQUAL TO 1 PREVENT A 0 CUT FOR TEXT
    if K == 1:
        text = text[1:-1]
    s = pattern + sentinel_char + text
    patLen = len(pattern)
    sLen = len(s)
    z = [0]*sLen
    left = right = 0
    for i in range(1,sLen):
        if i > right:
            left = right = i
            while right < sLen and s[right-left] == s[right]:
                right += 1
            z[i] = right - left
            right -= 1
        else:
            k = i - left
            if z[k] < right - i + 1:
                z[i] = z[k]
            else:
                left = i
                while right < sLen and s[right-left]==s[right]:
                    right += 1
                z[i]=right-left
                right-=1
        if z[i]==patLen: return "YES"
    return "NO"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process Input")
    # if no arguments specified it is assumed to be the submission input
    parser.add_argument('--validation', action=argparse.BooleanOptionalAction, default=False, help='specify if validation')
    parser.add_argument('--sample', action=argparse.BooleanOptionalAction, default=False, help='specify if sample')
    args = parser.parse_args()
    input_type = ''
    if args.validation:
        input_type = 'validation_'
    if args.sample:
        input_type = 'sample_'
    result = []
    with open(f'inputs/{problem}_{input_type}input.txt', 'r') as f:
        T = int(f.readline())
        for t in range(1,T+1):
            result.append(f'Case #{t}: {main()}')
    with open(f'outputs/{problem}_output.txt', 'w') as f:
        f.write('\n'.join(result))