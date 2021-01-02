import math


def mylog(x):
    try:
        res = math.log(x)
        return res
    except ValueError:
        return -math.inf
