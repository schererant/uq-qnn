from collections.abc import Callable
import tensorflow as tf
# Set random seeds for reproducibility

tf.random.set_seed(42)



def quartic_data(input_data):
    """ Create labels with quartic function.
    
    Args:
        input_data: tf array
    
    Returns:
        y: quartic function applied to input array
    """

    y = tf.convert_to_tensor(tf.pow(input_data, 4))

    return y

def qubic_data(input_data):
    """ Create labels with qubic function.
    
    Args:
        input_data: tf array
    
    Returns:
        y: quartic function applied to input array
    """

    y = tf.convert_to_tensor(tf.pow(input_data, 3))

    return y

def sinusoidal_data(input_data):
    """ Create labels with sinusoidal function.
    
    Args:
        input_data: tf array
    
    Returns:
        y: quartic function applied to input array
    """

    y = tf.convert_to_tensor(tf.math.sin(input_data))

    return y


def get_data(n_data: int =100, sigma_noise_1: float = 0.0, datafunction: Callable = quartic_data):
    """Define a function based toy regression dataset.

    Args:
      n_data: number of data points
      sigma_noise_1: injected sigma noise on targets
      datafunction: function to compute labels based on input data

    Returns:
      train_input, train_target, test_input, test_target
    """
    x_min = 0
    x_max = 1
    X_train = tf.linspace(x_min, x_max, n_data)
    
    # split training set
    gap_start = x_min + 0.35 * (x_max - x_min)
    gap_end = x_min + 0.6 * (x_max - x_min)

    # create label noise
    noise_1 = tf.random.normal([n_data], 0, 1, tf.float32, seed=1) * sigma_noise_1
    noise_1 = tf.where(X_train > gap_end, 0.0, noise_1)  # Only add noise to the left

 
    # create simple function based labels data set and
    # add gaussian noise
    label_noise = noise_1
    y_train = datafunction(X_train) # + label_noise

    #:TODO @nina: why do we need this?
    train_idx = (X_train < gap_end) & (X_train > gap_start)

    # update X_train
    X_train = X_train[~train_idx]
    y_train = y_train[~train_idx]

    # test over the whole line
    X_test = tf.linspace(x_min, x_max, 500) #TODO: @nina: why 500?
    y_test = datafunction(X_test)


    return X_train, y_train, X_test, y_test, label_noise