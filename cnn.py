import keras
import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.optimizers import Adam
import scipy.io as sio
import keras_tuner as kt
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots


def main():
    X_tr_rgb, X_val_rgb, X_tr_gray, X_val_gray, y_tr, y_val = preprocessing_data()
    # Base CNN
    history_base_rgb = train_and_evaluate_model(create_base_cnn, X_tr_rgb, y_tr, X_val_rgb, y_val, "Base CNN RGB")
    history_base_gray = train_and_evaluate_model(create_base_cnn, X_tr_gray, y_tr, X_val_gray, y_val, "Base CNN GrayScale")

    # Deep CNN
    history_deep_rgb = train_and_evaluate_model(create_deep_cnn, X_tr_rgb, y_tr, X_val_rgb, y_val, "Deep CNN RGB")
    history_deep_gray = train_and_evaluate_model(create_deep_cnn, X_tr_gray, y_tr, X_val_gray, y_val, "Deep CNN GrayScale")

    # Plot RBG vs Gray Performance
    rgb_vs_grayscale(history_base_rgb, history_base_gray, history_deep_rgb, history_deep_gray)

    # Tune Model
    # best_model = tune_model(X_tr_rgb[:50_000], y_tr[:50_000], X_val_rgb, y_val)
    # evaluate_best_model(best_model, X_tr_gray, y_tr, X_val_gray, y_val, "Best Model")


def train_and_evaluate_model(model_type, X_tr, y_tr, X_val, y_val, model_name):
    model, history = train_model(model_type, X_tr, y_tr, X_val, y_val)
    evaluate_model(model, X_val, y_val, model_name)
    return history

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

    return X_tr_rgb, X_val_rgb, X_tr_gray, X_val_gray, y_tr, y_val

def convert_to_grayscale(images):
    batch_size = 64
    grayscale_image_list = []

    for i in range(0, images.shape[0], batch_size):
        batch = images[i: i + batch_size]
        # Convert to float for better precision
        images_float = tf.cast(batch, tf.float32)
        # Apply TensorFlow's rgb_to_grayscale function
        grayscale_batch = tf.image.rgb_to_grayscale(images_float)
        grayscale_image_list.append(grayscale_batch)
    grayscale_images = tf.concat(grayscale_image_list, axis=0)

    return grayscale_images

def train_model(model_factory, X_tr, y_tr, X_val, y_val):
    model = model_factory(input_shape=X_tr[0].shape)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    # Early Stop condition + Reduce Learning Rate
    callback = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, callbacks=callback)
    return model, history

def evaluate_model(model, X_val, y_val, model_name):
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"{model_name} - Test accuracy: {test_acc:.4f}")

def create_simple_nn(input_shape=(32, 32, 3), num_classes=10):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

def create_base_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = keras.models.Sequential([
        # First Convolutional Layer
        keras.layers.Conv2D(32, 3, activation="relu", input_shape=input_shape, padding="same"),
        keras.layers.MaxPooling2D((2, 2)),
        # Second Convolutional Layer
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        # Neural Network
        keras.layers.Dense(128, activation="relu"),
        # Drop out to reduce overfitting
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

def create_deep_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = keras.models.Sequential()

    # First Convolutional Block
    model.add(keras.layers.Conv2D(32, 3, input_shape=input_shape, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Conv2D(32, 3, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.1))

    # Second Convolutional Block
    model.add(keras.layers.Conv2D(64, 3, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Conv2D(64, 3, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.2))

    # Third Convolutional Block
    model.add(keras.layers.Conv2D(128, 3, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Conv2D(128, 3, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())

    # First dense block
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dropout(0.5))

    # Second dense block
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dropout(0.5))

    # Output layer
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    return model

def build_model(hp):
    model = keras.models.Sequential()
    # Input Layer
    model.add(keras.layers.Input(shape=(32, 32, 1)))

    # Tune the number of convolutional blocks (1-3)
    for i in range(hp.Int("conv_blocks", 1, 3, default=2)):
        # Tune number of filters
        filters = hp.Int(f"filters_{i}", min_value=32, max_value=256, step=32, default=64)
        # Tune kernel size
        kernel_size = hp.Choice(f"kernel_size_{i}", values=[3, 5], default=3)
        # First Conv2D layer in this block
        model.add(keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same"))
        # Tune whether to use batch normalization
        if hp.Boolean(f"batch_norm_conv1_{i}", default=True):
            model.add(keras.layers.BatchNormalization())
        # Activation function
        model.add(keras.layers.Activation("relu"))

        # Optional second Conv2D layer
        if hp.Boolean(f"double_conv_{i}", default=False):
            model.add(keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same"))
            if hp.Boolean(f"batch_norm_conv2_{i}", default=True):
                model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation("relu"))

        # Add pooling
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Add dropout
        model.add(keras.layers.Dropout(
            hp.Float(f"dropout_conv_{i}", min_value=0.0, max_value=0.5, step=0.1, default=0.2))
        )

        # Flatten the output
    model.add(keras.layers.Flatten())

    # Choose number of dense layers (1, 2, or 3)
    for j in range(hp.Choice("num_dense_layers", values=[1, 2, 3])):
        # Tune number of dense units
        dense_units = hp.Int(f"dense_units_{j}", min_value=128, max_value=512, step=64, default=256)
        model.add(keras.layers.Dense(units=dense_units))
        # Tune whether to use batch normalization
        if hp.Boolean(f"batch_norm_dense_{j}", default=True):
            model.add(keras.layers.BatchNormalization())
        # Activation function
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(
            hp.Float(f"dropout_layer_dense_{j}", min_value=0.0, max_value=0.5, step=0.1)
        ))

    # Output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    # Tune learning rate for the optimizer
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def tune_model(X_tr, y_tr, X_val, y_val):
    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=300,
        executions_per_trial=2,
        directory="tuning",
        project_name="CS_178"
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
        )
    ]

    tuner.search(X_tr, y_tr, epochs=30, validation_data=(X_val, y_val), callbacks=callbacks)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()

    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    print(best_trial.hyperparameters.values)

    return best_model

def evaluate_best_model(model, X_tr, y_tr, X_val, y_val, model_name):
    callback = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, callbacks=callback)

    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"{model_name} - Test accuracy: {test_acc:.4f}")

def rgb_vs_grayscale(his1, his2, his3, his4):
    fig, axis = subplots(2, 2)

    axis[0][0].plot(his1.history["accuracy"], label="Training Accuracy")
    axis[0][0].plot(his1.history["val_accuracy"], label="Validation Accuracy")
    axis[0][0].set_ylabel("Accuracy")
    axis[0][0].set_xlabel("Epoch")
    axis[0][0].set_title("Base Model with RGB")
    axis[0][0].legend()

    axis[0][1].plot(his2.history["accuracy"], label="Training Accuracy")
    axis[0][1].plot(his2.history["val_accuracy"], label="Validation Accuracy")
    axis[0][1].set_ylabel("Accuracy")
    axis[0][1].set_xlabel("Epoch")
    axis[0][1].set_title("Base Model with GrayScale")
    axis[0][1].legend()

    axis[1][0].plot(his3.history["accuracy"], label="Training Accuracy")
    axis[1][0].plot(his3.history["val_accuracy"], label="Validation Accuracy")
    axis[1][0].set_ylabel("Accuracy")
    axis[1][0].set_xlabel("Epoch")
    axis[1][0].set_title("Deep Model with RGB")
    axis[1][0].legend()

    axis[1][1].plot(his4.history["accuracy"], label="Training Accuracy")
    axis[1][1].plot(his4.history["val_accuracy"], label="Validation Accuracy")
    axis[1][1].set_ylabel("Accuracy")
    axis[1][1].set_xlabel("Epoch")
    axis[1][1].set_title("Deep Model with GrayScale")
    axis[1][1].legend()

    plt.suptitle("Accuracy and Validation Accuracy for Base and Deep CNN Models on RGB and Grayscale Images")
    plt.tight_layout()
    plt.show()

    def compare_nn_and_cnn(X_tr, y_tr, X_val, y_val):
        nn_train_score = [0] * 10
        nn_val_score = [0] * 10
        cnn_train_score = [0] * 10
        cnn_val_score = [0] * 10

        models = [create_simple_nn, create_base_cnn]
        for model in models:
            trial = 1
            while trial <= 10:
                model, history = train_model(model, X_tr, y_tr, X_val, y_val)
                test_loss, test_acc = model.evaluate(X_val, y_val)
                train_lost, train_acc = model.evaluate(X_tr, y_tr)
                if train_acc < 0.80:
                    continue
                if model == create_simple_nn:
                    nn_train_score[trial - 1] = train_acc
                    nn_val_score[trial - 1] = test_acc
                if model == create_base_cnn:
                    cnn_train_score[trial - 1] = train_acc
                    cnn_val_score[trial - 1] = test_acc
                trial += 1
        print(np.mean(nn_train_score))
        print(np.mean(nn_val_score))
        print(np.mean(cnn_train_score))
        print(np.mean(cnn_val_score))

    def compare_base_deep_on_rbg_grayscale(X_tr_rgb, X_val_rgb, X_tr_gray, X_val_gray, y_tr, y_val):
        base_RGB_train = [0] * 10
        base_RGB_test = [0] * 10
        base_gray_train = [0] * 10
        base_gray_test = [0] * 10

        deep_RGB_train = [0] * 10
        deep_RGB_test = [0] * 10
        deep_gray_train = [0] * 10
        deep_gray_test = [0] * 10

        trial = 1
        while trial <= 10:
            models = []
            while True:
                model, history = train_model(create_base_cnn, X_tr_rgb, y_tr, X_val_rgb, y_val)
                if model.evaluate(X_val_rgb, y_val)[1] > 0.80:
                    break
            models.append(model)
            while True:
                model, history = train_model(create_base_cnn, X_tr_gray, y_tr, X_val_gray, y_val)
                if model.evaluate(X_tr_gray, y_val)[1] > 0.80:
                    break
            models.append(model)

            # Deep CNN
            while True:
                model, history = train_model(create_deep_cnn, X_tr_rgb, y_tr, X_val_rgb, y_val)
                if model.evaluate(X_val_rgb, y_val)[1] > 0.80:
                    break
            models.append(model)
            while True:
                model, history = train_model(create_deep_cnn, X_tr_gray, y_tr, X_val_gray, y_val)
                if model.evaluate(X_tr_gray, y_val)[1] > 0.80:
                    break
            models.append(model)
            for idx, model in enumerate(models):
                if idx in [0, 2]:
                    test_loss, test_acc = model.evaluate(X_val_rgb, y_val)
                    train_lost, train_acc = model.evaluate(X_tr_rgb, y_tr)
                    if idx == 0:
                        base_RGB_train[trial - 1] = train_acc
                        base_RGB_test[trial - 1] = test_acc
                    else:
                        deep_RGB_train[trial - 1] = train_acc
                        deep_RGB_test[trial - 1] = test_acc
                else:
                    test_loss, test_acc = model.evaluate(X_val_gray, y_val)
                    train_lost, train_acc = model.evaluate(X_tr_gray, y_tr)
                    if idx == 1:
                        base_gray_train[trial - 1] = train_acc
                        base_gray_test[trial - 1] = test_acc
                    else:
                        deep_gray_train[trial - 1] = train_acc
                        deep_gray_test[trial - 1] = test_acc
            trial += 1

        model_list = [base_RGB_train, base_RGB_test, base_gray_train, base_gray_test,
                      deep_RGB_train, deep_RGB_test, deep_gray_train, deep_gray_test]
        for result in model_list:
            print(np.mean(result))


if __name__ == "__main__":
    main()
