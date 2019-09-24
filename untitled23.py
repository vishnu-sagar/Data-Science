arranging coins

import math
def arrangeCoins(n):
    return int((math.sqrt(8*n+1)-1) / 2)

print(arrangeCoins(3))


