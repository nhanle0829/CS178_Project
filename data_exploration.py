import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

data = sio.loadmat("train_32x32.mat")
X = data['X']
y = data['y']
y[y == 10] = 0


def random_indexes_number(number, number_of_index=5):
    indices = np.where(y == number)[0]
    if len(indices) > number_of_index:
        indices = np.random.choice(indices, size=5, replace=False)
    return indices

