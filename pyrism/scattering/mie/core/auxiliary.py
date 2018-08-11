import numpy as np


def end_sum(A0, A1, num):
    stop = True
    pDiff = np.abs((A1 - A0) / A0) * 100

    try:
        for t in range(num):
            if pDiff[t] >= 0.001 or A0[t] == 0:
                stop = False
            else:
                pass
        return stop
    except:
        if pDiff >= 0.001 or A0 == 0:
            stop = False
        else:
            pass
        return stop
