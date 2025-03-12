import keras
import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.optimizers import Adam
import scipy.io as sio
from keras.src.utils import to_categorical


def main():
    X_tr_rgb, X_val_rgb, X_tr_gray, X_val_gray, y_tr, y_val = preprocessing_data()
    # Base CNN
    train_and_evaluate_model(create_base_cnn, X_tr_rgb, y_tr, X_val_rgb, y_val, "Base CNN RGB")
    train_and_evaluate_model(create_base_cnn, X_tr_gray, y_tr, X_val_gray, y_val, "Base CNN GrayScale")

def train_and_evaluate_model(model_type, X_tr, y_tr, X_val, y_val, model_name):
    model = train_model(model_type, X_tr, y_tr, X_val, y_val)
    evaluate_model(model, X_val, y_val, model_name)

def preprocessing_data():
    # Load data
    data_tr = sio.loadmat("train_32x32.mat")
    X_tr = data_tr['X']  # Shape would be (32, 32, 3, num_samples)
    y_tr = data_tr['y']  # Class labels

    data_val = sio.loadmat("test_32x32.mat")
    X_val = data_val['X']
    y_val = data_val['y']

    # Preprocessing
    # RGB Images
    X_tr_rgb = X_tr.transpose(3, 0, 1, 2)
    X_val_rgb = X_val.transpose(3, 0, 1, 2)

    # GrayScale Image
    X_tr_gray = convert_to_grayscale(X_tr_rgb)
    X_val_gray = convert_to_grayscale(X_val_rgb)

    y_tr[y_tr == 10] = 0
    y_val[y_val == 10] = 0

    # One Hot Encoding
    y_tr = to_categorical(y_tr, num_classes=10)
    y_val = to_categorical(y_val, num_classes=10)

    return X_tr_rgb, X_val_rgb, X_tr_gray, X_val_gray, y_tr, y_val

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
    model = model_factory(input_shape=X_tr[0].shape)
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
    return model

def evaluate_model(model, X_val, y_val, model_name):
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"{model_name} - Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
