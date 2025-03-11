import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

df = pd.read_csv("legend.csv")
# print(df.head())
df["emotion"] = df["emotion"].str.capitalize()
df["emotion"] = df["emotion"].replace("Contempt", "Disgust")
emotion_counts = df["emotion"].value_counts()

plt.figure(figsize=(10, 10))
plt.title('Distribution of Emotions in Dataset')
plt.xlabel('Emotion')
plt.ylabel('Count')
emotion_counts.plot(kind="bar")
plt.show()

images_dir = "./images"
for i, row in df.iloc[:5].iterrows():
    img_path = os.path.join(images_dir, row["image"])
    try:
        img = Image.open(img_path)

        # Plot the sample image
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"Emotion: {row['emotion']}")
        plt.axis('off')
        plt.show()

        # Plot pixel intensity distribution
        img_array = np.array(img)
        plt.figure(figsize=(8, 3))
        plt.hist(img_array.flatten(), bins=50)
        plt.title('Pixel Intensity Distribution')
        plt.show()

    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
