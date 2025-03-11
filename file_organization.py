import pandas as pd
from pathlib import Path
import shutil

df = pd.read_csv("legend.csv")
# print(df.head())
df["emotion"] = df["emotion"].str.capitalize()
df["emotion"] = df["emotion"].replace("Contempt", "Disgust")
print(df["emotion"].value_counts())


data_path = "./images"
dest_path = "./emotion_dataset"
Path(dest_path).mkdir(exist_ok=True)

for emotion in df["emotion"].unique():
    new_dir = Path(dest_path) / emotion
    new_dir.mkdir(exist_ok=True)

# Copy image
for idx in range(len(df)):
    shutil.copy(data_path + '/' + df["image"][idx], dest_path + '/' + df["emotion"][idx] + '/' + df["image"][idx])
    if Path(df["image"][idx]).suffix.upper() not in [".JPG", ".JPEG", ".PNG"]:
        print(df["image"][idx])