import tensorflow as tf
import numpy as np

###### MLP BASELINE ######

#TODO: adapt hidden layer,  epochs, learning rate
def train_mlp_baseline(X_train, y_train, hidden_layers=[64, 64], epochs=100, learning_rate=0.01):
    """Train a simple MLP baseline model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(1,)))
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    return model

def predict_mlp_baseline(model, X_test):
    """Predict using the trained MLP baseline model."""
    return model.predict(X_test).flatten()



###### POLYNOMIAL BASELINE ######

def train_polynomial_baseline(X_train, y_train, degree=3):
    """Fit a polynomial baseline model."""
    coeffs = np.polyfit(X_train.numpy(), y_train.numpy(), degree)
    return coeffs

def predict_polynomial_baseline(coeffs, X_test):
    """Predict using the fitted polynomial baseline model."""
    poly = np.poly1d(coeffs)
    return poly(X_test.numpy())