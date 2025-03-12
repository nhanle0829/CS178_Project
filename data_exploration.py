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

# Samples for each digit
fig, axes = plt.subplots(10, 5, figsize=(20, 10))

for i in range(10):
    random_indices = random_indexes_number(i)
    for j in range(5):
        img = X[:, :, :, random_indices[j]]
        axes[i][j].imshow(img)

        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])

plt.suptitle("SVHN Dataset Samples for Each Digit", fontsize=30)
plt.tight_layout()
plt.show()

# Plot class distribution
unique, count = np.unique(y, return_counts=True)
plt.bar(unique, count)
plt.xlabel("Digit")
plt.ylabel("Count")
plt.title("Class Distribution in SVHN Dataset")
plt.xticks(unique)
plt.show()