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
def plat_samples_for_each_digit():
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
def plot_class_distribution():
    unique, count = np.unique(y, return_counts=True)
    plt.bar(unique, count)
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.title("Class Distribution in SVHN Dataset")
    plt.xticks(unique)
    plt.show()

# Plot image RGB channel
def image_rgb_channel():
    red = X[:, :, 0, 0]
    blue = X[:, :, 1, 0]
    green = X[:, :, 2, 0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(red, cmap="Reds")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(blue, cmap="Blues")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(green, cmap="Greens")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.suptitle("Image and its individual RGB channels")
    plt.tight_layout()
    plt.show()

# Plot pixel distribution
def plot_pixel_distribution(sample_size=1000):
    X_reshaped = X.transpose(3, 0, 1, 2)
    if len(X_reshaped[0] > sample_size):
        indices = np.random.choice(X_reshaped.shape[0], size=sample_size, replace=False)
        X_reshaped = X_reshaped[indices]

    pixel_r = X_reshaped[:, :, :, 0].flatten()
    pixel_g = X_reshaped[:, :, :, 1].flatten()
    pixel_b = X_reshaped[:, :, :, 2].flatten()
    pixel_list = [pixel_r, pixel_g, pixel_b]


    fig, axes = plt.subplots(1, 3)
    axes[0].hist(pixel_list[0], bins=50, color="r")
    axes[1].hist(pixel_list[1], bins=50, color="g")
    axes[2].hist(pixel_list[2], bins=50, color="b")

    plt.suptitle("Pixel Distribution")
    plt.tight_layout()
    plt.show()

def image_after_normalization():


if __name__ == "__main__":
    print(f"Single image shape: {X[:, :, :, 0].shape}")
    
    plat_samples_for_each_digit()
    plot_class_distribution()
    image_rgb_channel()
    plot_pixel_distribution()