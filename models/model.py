
import numpy as np
import yaml



with open(r'config\config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

SEED = config['SEED']
np.random.seed(SEED) # Set random seed.
batch_size = config['batch_size'] # Set batch size.
alpha = config['momentum'] # set alpha for momentum.


class NNLayer:
    """
    Fully conected layer.

    Args:
        n_inputs: number of inputs.
        n_neurons: number of neurons (that shows the number of outputs of layer).
        w_param: a list of parameters that needs for weight initialization (bias, mean of weights, std of weights).
    """

    def __init__(self, n_inputs, n_neurons, w_param):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.w_pre = 0
        self.b_pre = 0
        self.weight_init(w_param)

    def weight_init(self, w_param):
        """
        Weight initialization.
        Set biases and weights.

        Args:
            w_param: a list of parameters that needs for weight initialization 
            (bias, mean of weights, std of weights).
        """

        bias, w_mean, w_std = w_param
        # self.biases = np.full((1, self.n_neurons), bias, dtype=np.float32)
        self.biases = np.full((self.n_neurons), bias, dtype=np.float32)
        self.weights = np.random.normal(w_mean, w_std, (self.n_inputs, self.n_neurons))
        print("init bias and w shape",np.shape(self.biases), np.shape(self.weights))

    def forward(self, input):
        self.input  = input
        self.out = np.dot(self.input, self.weights) + self.biases
        return self.out

    def backward(self, dz, lr):
        """
        Back propagation.
        Args:
            dz: dL/dz
            lr: learning rate

        Returns:
            dx: DL/dx
        """

        self.dx = np.dot(dz, self.weights.T)
        self.dw = np.dot(self.input.T, dz) # dw: DL/dw
        self.db = np.full(dz.shape[1], 1) # db: DL/db

        # update parameters
        
        # # """Stochastic Gradient Descent."""
        # self.weights -= lr * self.dw #weights_error
        # self.biases -= lr * self.db # output_error

        # """SGD + Momentum."""
        self.w_pre = alpha * self.w_pre  - lr * self.dw
        self.b_pre = alpha * self.b_pre  - lr * self.db
        self.weights += self.w_pre
        self.biases += self.b_pre

        return self.dx


class ReLU:
    """
    ReLU activation function.
    """

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self, dz, lr):
        max_in = np.argmax(self.input, axis=1)
        dx = np.zeros_like(self.input)
        for i in range(len(self.input)):
            dx[i, max_in[i]] = dz[i, max_in[i]]

        return dx


class Softmax:
    """
    Softmax activation function.
    """

    def forward(self, input):
        self.input = input
        exp_x = np.exp(self.input - np.max(self.input, axis=1, keepdims=True)) # Computationally stable softmax
        self.sum = np.sum(exp_x, axis=1, keepdims=True)
        self.output = exp_x / self.sum
        return self.output

    def backward(self, dz, lr):
        return dz

    
class CreateModel:
    """
    Create nn model contaning an input layer, diesired hiden layer and an output layer.
    activation function for hidden layers: ReLU.
    activation function for output layers: Softmax.

    Args:
        n_layers: number of hidden layers.
        n_input: number of inputs.
        n_neurons: a list where each element shows the number of neurons in that layer.
        n_cls: number of classes.
        w_p: weight initialization parameters.
    """

    def __init__(self, n_layers:int, n_input:int, n_neurons:list, n_cls:int, w_p:list) -> None:
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.n_neurons = np.append(n_input, n_neurons)
        self.w_p = w_p
        self.layers = []
        if len(n_neurons) != n_layers:
            raise Exception(f"You have {self.n_layers} layer(s) but not set number of neurons for all of them.")

        self.create_nn()

    def create_nn(self):
        for i in range(self.n_layers):
            self.layers.append(NNLayer(self.n_neurons[i], self.n_neurons[i+1], self.w_p))
            self.layers.append(ReLU())
        self.layers.append(NNLayer(self.n_neurons[-1], self.n_cls, self.w_p)) # output layer
        self.layers.append(Softmax())

        return self.layers

