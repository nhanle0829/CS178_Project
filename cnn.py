import keras
import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.optimizers import Adam
import scipy.io as sio
from keras.src.utils import to_categorical

# Preprocessing
X_tr = X_tr.transpose(3, 0, 1, 2)
X_val = X_val.transpose(3, 0, 1, 2)

X_tr = X_tr.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

if np.min(y_tr) == 1:
    y_tr = y_tr - 1
    y_val = y_val - 1

y_tr = to_categorical(y_tr, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)

def main():
    pass

def preprocessing_data():
    # Load data
    data_tr = sio.loadmat("train_32x32.mat")
    X_tr = data_tr['X']  # Shape would be (32, 32, 3, num_samples)
    y_tr = data_tr['y']  # Class labels

    data_val = sio.loadmat("test_32x32.mat")
    X_val = data_val['X']
    y_val = data_val['y']

    # Preprocessing
    X_tr_rgb = X_tr.transpose(3, 0, 1, 2)
    X_val_rgb = X_val.transpose(3, 0, 1, 2)

def convert_to_grayscale(images):
    # Convert to float for better precision
    images_float = tf.cast(images, tf.float32)

    # Apply TensorFlow's rgb_to_grayscale function
    grayscale_images = tf.image.rgb_to_grayscale(images_float)

    return grayscale_images

def create_base_cnn(input_shape=(32, 32, 3), num_classes=10):
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

def train_model(model_factory, X_tr, y_tr, X_val, y_val):
    model = model_factory()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Early Stop condition + Reduce Learning Rate
    callback = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, callbacks=callback)

def evaluate_model(model):
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"Base_CNN - Test accuracy: {test_acc:.4f}")
