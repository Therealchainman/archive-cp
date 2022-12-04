

```py
with open('test_data/test_set_2/ts2_input.txt', 'r') as f, open('output.txt', 'w') as out:

print('here', file = out)

from sys import *
setrecursionlimit(int(1e6))

import faulthandler
faulthandler.enable()

```