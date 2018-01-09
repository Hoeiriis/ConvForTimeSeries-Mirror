import numpy as np
import keras.layers as kl

def Conv1D_input_layer(X_model_input, n_filters, kernel_shape):

    # Input 1
    X_model_1 = kl.Conv1D(n_filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(X_model_input)
    X_model_1 = kl.BatchNormalization()(X_model_1)

    # Input 2
    X_model_2 = kl.Conv1D(n_filters, kernel_size=int(kernel_shape), padding="same", kernel_initializer="he_normal")(X_model_input)
    X_model_2 = kl.BatchNormalization()(X_model_2)

    # Combination
    X_model = kl.Concatenate()([X_model_1, X_model_2])
    X_model = kl.Activation("relu")(X_model)
    X_model = kl.MaxPool1D()(X_model)

    return X_model

def Conv1D_identity_block(X_model, n_filters, kernel_shape):

    X_model_original = X_model

    filter1, filter2 = n_filters

    # First layer
    X_model = kl.Conv1D(filter1, 1, padding="valid", kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)
    X_model = kl.Activation("relu")(X_model)

    # Second layer
    X_model = kl.Conv1D(filter2, int(kernel_shape), padding="same", kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)
    X_model = kl.Activation("relu")(X_model)

    # Third layer
    X_model = kl.Conv1D(filter2, 1, padding="valid", kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)

    # Final
    X_model = kl.Add()([X_model, X_model_original])
    X_model = kl.Activation("relu")(X_model)

    return X_model

def Conv1D_res_block(X_model, n_filters, kernel_shape, stride=2):
    filter1, filter2, filter3 = n_filters

    X_model_original = X_model

    # First layer
    X_model = kl.Conv1D(filter1, kernel_size=1, strides=stride, padding="same", kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)
    X_model = kl.Activation("relu")(X_model)

    # Second layer
    X_model = kl.Conv1D(filter2, kernel_size=int(kernel_shape), padding="same", kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)
    X_model = kl.Activation("relu")(X_model)

    # Third layer
    X_model = kl.Conv1D(filter3, kernel_size=1, padding="valid", kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)

    # Layer in original
    X_model_original = kl.Conv1D(filter3, kernel_size=1, strides=stride,
                                 padding="valid", kernel_initializer="he_normal")(X_model_original)
    X_model_original = kl.BatchNormalization()(X_model_original)

    # Final
    X_model = kl.Add()([X_model, X_model_original])
    X_model = kl.Activation("relu")(X_model)

    return X_model