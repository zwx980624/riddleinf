import numpy as np

def MRR(pos, n=5):
    ret = 1.0 / (pos+1) if pos <= n else 0
    return ret

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x