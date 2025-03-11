import pandas as pd
from pathlib import Path

df = pd.read_csv("legend.csv")
# print(df.head())
df["emotion"] = df["emotion"].str.capitalize()
df["emotion"] = df["emotion"].replace("Contempt", "Disgust")
print(df["emotion"].value_counts())

data_path = Path('.')
dest_path = Path("./emotion_dataset")
dest_path.mkdir(exist_ok=True)