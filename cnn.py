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

def create_base_cnn(input_shape, num_classes):
    base_cnn = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return base_cnn

