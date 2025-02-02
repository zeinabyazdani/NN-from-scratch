
import yaml
import sys
sys.path.append("train")
import train

# Set random seed.
with open(r'config\config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

loss_function = config['loss']


def testing(nn_model, x_test, y_test):
    """
    Test NN model.

    Args:
        nn_model: NN model
        x_test(numpy array): test input

    Returns:
        y_hat: Model predictions for test data
    """

    y_hat = x_test
    for layer in nn_model.layers:
        y_hat = layer.forward(y_hat)


    # Calculate loss
    if loss_function.lower() == 'ce':
        loss_te = train.cross_entropy_loss(y_test, y_hat)[0]
    elif loss_function.lower() == 'mse':
        loss_te = train.mse_loss(y_test, y_hat)[0]
    else: raise Exception("Undefined loss")

    return y_hat, loss_te

