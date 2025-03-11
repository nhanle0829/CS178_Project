import keras

img_height, img_width = 48, 48
batch_size = 32

# Load data
train_ds = keras.utils.image_dataset_from_directory(
    "emotion_dataset",
    validation_split=0.2,
    label_mode="categorical",
    color_mode="grayscale",
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    "emotion_dataset",
    validation_split=0.2,
    label_mode="categorical",
    color_mode="grayscale",
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Preprocessing
train_datagen = keras.Sequential([
    keras.layers.Rescaling(1./255),
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomContrast(0.2)
])

train_ds = train_ds.map(lambda x, y: (train_datagen(x, training=True), y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

print(train_ds.class_names)