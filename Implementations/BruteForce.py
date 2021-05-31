from __future__ import print_function

import itertools
import sys

N = int(sys.stdin.readline())
T = int(sys.stdin.readline())
vals = list(map(int, sys.stdin.readlines()))

for i in range(1, N + 1):
    for subset in itertools.combinations(vals, i):
        if sum(subset) == 0:
            print("Found")
            sys.exit()
print("None")