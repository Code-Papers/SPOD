import numpy as np

def constfn(val):
    def f(a):
        return val * a
    return f
        
if __name__ == "__main__":
    lr = 3e-4
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    print('yes')
    for i in range(1, 11):
        frac = 1.0 - (i - 1.0) / 10
        lrnow = lr(frac)
        print(lrnow, (i-1.0)/10)

    