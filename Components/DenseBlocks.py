import numpy as np
import keras.layers as kl

def dense_simple_layer(X_model, hidden_units):
    """
    Simple dense layer.
    :param X_model: Model input
    :param hidden_units: amount of inputs in hidden layer.
    :return: Model output
    """

    X_model = kl.Dense(hidden_units, kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)
    X_model = kl.Activation("relu")(X_model)

    return X_model

def dense_identityblock(X_model, hidden_units, last_hidden_units):
    """
    Input and output must match
    :param X_model: Model input
    :param hidden_units: amount of inputs in hidden layers
    :param last_hidden_units: amount of inputs in the last hidden layer, this should be the same as input
    :return: Model output
    """

    X_model_original = X_model

    # First layer
    X_model = kl.Dense(hidden_units, kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)
    X_model = kl.Activation("relu")(X_model)

    # Second layer
    X_model = kl.Dense(hidden_units, kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)
    X_model = kl.Activation("relu")(X_model)

    # Third layer
    X_model = kl.Dense(last_hidden_units, kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)

    # Final
    X_model = kl.Add()([X_model, X_model_original])
    X_model = kl.Activation("relu")(X_model)

    return X_model


def dense_res_block(X_model, hidden_units):
    """
    For when input and output shape do not match
    :param X_model: Model input
    :param hidden_units: amount of inputs in hidden layers
    :return: Model output
    """

    X_model_original = X_model

    # First layer
    X_model = kl.Dense(hidden_units, kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)
    X_model = kl.Activation("relu")(X_model)

    # Second layer
    X_model = kl.Dense(hidden_units, kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)
    X_model = kl.Activation("relu")(X_model)

    # Third layer
    X_model = kl.Dense(hidden_units, kernel_initializer="he_normal")(X_model)
    X_model = kl.BatchNormalization()(X_model)

    # Layer in original
    X_model_original = kl.Dense(hidden_units, kernel_initializer="he_normal")(X_model_original)
    X_model_original = kl.BatchNormalization()(X_model_original)

    # Final
    X_model = kl.Add()([X_model, X_model_original])
    X_model = kl.Activation("relu")(X_model)

    return X_model