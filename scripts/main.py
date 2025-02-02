
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
sys.path.append("data")
sys.path.append("models")
sys.path.append("utils")
import data_loader
import model
import train
import evaluate
import visualization
import metrics


#########################################################
################  Read and set configs  #################
#########################################################

with open(r'config\config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Random seed
SEED = config['SEED']

# Access hyperparameters
num_layers      = config['num_layers']
num_neurons     = config['num_neurons']
batch_size      = config['batch_size']
learning_rate   = config['learning_rate']
num_epochs      = config['num_epochs']
momentum        = config['momentum']
loss            = config['loss']
normalization   = config['normalization']

# Access dataset parameters
path_train       = config['dataset']['path_train']
path_test        = config['dataset']['path_test']
file_train_names = config['dataset']['file_train_names']
file_test_names  = config['dataset']['file_test_names']
num_classes      = config['dataset']['num_classes']
train_split      = config['dataset']['train_split']

# Access Weight initialization parameters
bias   = config['bias']
w_mean = config['w_mean']
w_std  = config['w_std']


#########################################################
################  Data preprocessing   ##################
#########################################################

# Load data
# partition=True If you want to work on a small part of the data (Including 1000 samples)
x_train, y_train = data_loader.load_data(path_train, file_train_names, partition=False) 
x_test, y_test   = data_loader.load_data(path_test, file_test_names)

print("input data type (train ad test):", type(x_train), type(y_train)) # <class 'list'> <class 'numpy.ndarray'>
print("input data length (train ad test):", len(x_train), len(y_train))

# Display images
data_loader.show_images(x_train, y_train)

# Plot histogram of data
data_loader.histogram(y_train)

# Resize images
x_tr_re = data_loader.resizing(x_train)
x_te_re = data_loader.resizing(x_test)
print("resized data type (train):", type(x_tr_re))
print("resized data shape (train, test):", np.shape(x_tr_re), np.shape(x_te_re))
data_loader.show_images(x_tr_re, y_train)

# Flat data
x_tr = data_loader.flatten(x_tr_re)
x_te = data_loader.flatten(x_te_re)
print("flat data type (train):", type(x_tr))
print("flat data shape (train, test):", np.shape(x_tr), np.shape(x_te))

# Data normalization
if normalization == 'std':
    x_tr = data_loader.normalization(x_tr)
    x_te = data_loader.normalization(x_te)

# One-hot encoding
y_tr_c = data_loader.to_categorical(y_train)
print("shape of train labels after one-hot encoding", y_tr_c.shape)

# Train and Validation data split
x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr_c, test_size=train_split, random_state=SEED)
print("input shapes (train, validation):", np.shape(x_tr), np.shape(x_val))
print("output shapes (train, validation):", np.shape(y_tr), np.shape(y_val))


#########################################################
################    train and test     ##################
#########################################################

# train
w_param = [bias, w_mean, w_std]
nn_model = model.CreateModel(num_layers, x_tr.shape[1], num_neurons, num_classes, w_param)
print("layers",nn_model.layers)
trian = train.Train(nn_model, loss)
history = trian.training(x_tr, y_tr, x_val, y_val, batch_size, num_epochs, learning_rate)

# Visualize result on training and validation
visualization.plot_history(history)

# Test
y_pred, loss_test = evaluate.testing(nn_model, x_te, data_loader.to_categorical(y_test))
y_pred = np.argmax(y_pred, axis=1)

print("test loss:", loss_test)
metrics.report_result(y_test, y_pred)
