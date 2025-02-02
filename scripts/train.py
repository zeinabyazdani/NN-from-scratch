
import numpy as np
from sklearn.metrics import accuracy_score


def mse_loss(y:np.array, y_hat:np.array) -> None:
    """
    Mean Squared Error loss function.

    Arge:
        y: True labels
        y_hat: Predictions (Output of forward path)

    Returns:
        mse: MSE loss
        mse_diff: Derivative of loss
    """

    # mse = np.mean(np.power(self.y_hat - self.y, 2))
    mse = np.mean(np.power(y_hat - y, 2))
    mse_diff = 2 * (y_hat - y) / (y).shape[0]
    # mse_diff = y_hat - y
                    
    return mse, mse_diff


def cross_entropy_loss(y:np.array, y_hat:np.array) -> None:
    """
    Cross Entropy loss function.

    Arge:
        y: True labels
        y_hat: Predictions (Output of forward path)

    Returns:
        ce: Cross Entropy loss
        ce_diff: Derivative of loss
    """

    ce = np.sum(-y * np.log2(y_hat)) / len(y)
    ce_diff = (y_hat - y)

    return ce, ce_diff
    


class Train:
    """
    Train process.

    Args:
        nn_model: neural network model.
    """

    def __init__(self, nn_model, loss_function) -> None:
        self.nn_model = nn_model
        self.loss_function = loss_function
        
    def training(self, x_train:np.array, y_train:np.array, x_val:np.array, y_val:np.array, batch_size:int, epoch:int, learning_rate:float):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val       
        self.batch_size = batch_size
        self.epochs = epoch
        self.learning_rate = learning_rate
        self.training_loss = []
        self.training_acc  = []
        self.validation_loss = []
        self.validation_acc  = []
        self.num_batches = len(x_train) // batch_size


        for ep in range(self.epochs):
            loss_epoch = 0
            y_hat_prob = []

            for i in range(0, self.x_train.shape[0], self.batch_size):

                # forward propagation
                output = self.x_train[i:i+self.batch_size]
                for layer in self.nn_model.layers:
                    output = layer.forward(output)
                y_hat_prob[i:i+self.batch_size] = output

                # compute loss
                if self.loss_function.lower() == 'ce':
                    loss = cross_entropy_loss(self.y_train[i:i+self.batch_size], output)
                elif self.loss_function.lower() == 'mse':
                    loss = mse_loss(self.y_train[i:i+self.batch_size], output)
                else: raise Exception("Undefined loss")
            
                loss_epoch += loss[0] # loss[0]: loss_batch
                
                # backward propagation
                dz = loss[1] # loss[1]: loss_batch_diff
                for layer in reversed(self.nn_model.layers):
                    dz = layer.backward(dz, self.learning_rate)


            # calculate epoch loss.
            loss_epoch /= self.num_batches          # average loss on batches

            # calculate epoch accuracy score.
            y_hat = np.argmax(y_hat_prob, axis=1)   # predicted class
            acc_epoch = accuracy_score(np.argmax(self.y_train, axis=1), y_hat)


            # Validation
            val_out = self.x_val
            # Forward path
            for layer in self.nn_model.layers:
                val_out = layer.forward(val_out)
            # Calculate loss
            if self.loss_function.lower() == 'ce':
                loss_val = cross_entropy_loss(y_val, val_out)[0]
            elif self.loss_function.lower() == 'mse':
                loss_val = mse_loss(y_val, val_out)[0]
            else: raise Exception("Undefined loss")

            # Calculate accuracy
            acc_val = accuracy_score(np.argmax(self.y_val, axis=1), np.argmax(val_out, axis=1))

            # save results of train an epoch.
            self.training_loss.append(loss_epoch)
            self.training_acc.append(acc_epoch)
            self.validation_loss.append(loss_val)
            self.validation_acc.append(acc_val)

            # print results of this epoch
            print(f"epoch {ep+1} / {self.epochs} \t\t loss = {loss_epoch:.4f} \t acc = {acc_epoch:.4f} \t\t val loss = {loss_val:.4f} \t acc = {acc_val:.4f}")

        return (self.training_loss, self.training_acc, self.validation_loss, self.validation_acc)
