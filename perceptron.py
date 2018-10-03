import numpy as np

def perceptron_sgd(X, Y):
    w = np.zeros(len(X[0]))
    eta = 1
    b = 0
    change = 1
    no_of_epoch = 200

    while change and no_of_epoch != 0:
        change = 0
        for i, x in enumerate(X):
            yin = np.dot(X[i], w) + b
            if yin > 0:
                yin = 1
            elif yin < 0:
                yin = -1
            else:
                yin = 0
            if yin != Y[i] and np.count_nonzero(X[i]) != 0:
                w = w + eta*X[i]*Y[i]
                b = b + eta*Y[i]
                change = 1
        no_of_epoch = no_of_epoch-1
    return w,b

'''
w,b = perceptron_sgd(X,y)
yin = np.dot(X[0], w) + b
print(yin)
print(w,b)
'''