import numpy as np
shape = [128, 7]


def data_gen():
    x = np.random.uniform(low=0, high=1, size=shape)
    w = np.array([1, 2, 3, 4, 5, 6, 7])
    y = np.dot(x, w) + np.random.uniform(low=0, high=0.1, size=(128,))
    return x, y


def ptxt_LinR(x, y):
    rate = 0.005
    w = np.zeros((shape[1]+1,))
    x = np.column_stack((x, np.zeros((shape[0],))+1))
    for epoch in range(10000):
        gradient = np.dot(np.dot(x, w) - y, x)
        w = w - rate * gradient
    return w


if __name__ == '__main__':
    x, y = data_gen()
    w = ptxt_LinR(x, y)
    print(w)
