import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)
# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Xavier initialization function
def xavier_init(shape):
    fan_in, fan_out = shape
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

def he_init(shape):
    fan_in = shape[0]  # Number of input units in the layer
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std

# Neural Network implementation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        # Xavier initialization for weights
        self.w1 = he_init((input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        print('W=',self.w1.shape)
        print('b=',self.b1.shape)
  
        self.w2 = he_init((hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
        print('W=',self.w2.shape)
        print('b=',self.b2.shape)
        
        self.learning_rate = learning_rate

    def forward(self, x):
        # First layer
        print('forward')

        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = relu(self.z1)
        print(self.a1.shape)

        # Second layer
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)
        print(self.a2.shape)
        return self.a2
         
    def compute_loss(self, y_true, y_pred):
        # https://numpy.org/doc/2.3/reference/generated/numpy.clip.html
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_loss_derivative(self, y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    def backward(self, x, y):
        m = y.shape[0]
        y = y.reshape(-1, 1)

        # Compute derivatives
        dL_da2 = self.compute_loss_derivative(y, self.a2)
        da2_dz2 = sigmoid_derivative(self.z2)
        dz2_dw2 = np.dot(self.a1.T, dL_da2 * da2_dz2) / m
        dz2_db2 = np.sum(dL_da2 * da2_dz2, axis=0, keepdims=True) / m

        dL_da1 = np.dot(dL_da2 * da2_dz2, self.w2.T)
        da1_dz1 = relu_derivative(self.z1)
        dz1_dw1 = np.dot(x.T, dL_da1 * da1_dz1) / m
        dz1_db1 = np.sum(dL_da1 * da1_dz1, axis=0, keepdims=True) / m

        return dz1_dw1, dz1_db1, dz2_dw2, dz2_db2

    def update_params_sgd(self, dw1, db1, dw2, db2):
        # Update weights and biases using SGD
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2

    def predict(self, x):
        return self.forward(x)

    def test_accuracy(self, x_test, y_test):
        predictions = self.predict(x_test)
        predictions = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions == y_test.reshape(-1, 1))
        return accuracy

    def train(self, x_train, y_train, epochs, batch_size=10):
        for epoch in range(epochs):
            print ('epoch = ', epoch)

            for i in range(0, x_train.shape[0], batch_size):
                print('----')
                print (i)
                #print (batch_size)

                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                print(x_batch.shape)
                print(x_batch)
                print(y_batch.shape)
                print(y_batch)

                y_pred = self.forward(x_batch)
                print ('prediction=', y_pred)

                loss = self.compute_loss(y_batch, y_pred)
                print ('loss=', loss)
                gradients = self.backward(x_batch, y_batch)

                print('gradients=')
                print('W1 = ',gradients[0].shape)
                print('b1 = ',gradients[1].shape)
                print('b1 = ',gradients[1])
                print('W2 = ',gradients[2].shape)
                print('b2 = ',gradients[3].shape)
                print('b2 = ',gradients[3])

                #dz1_dw1, dz1_db1, dz2_dw2, dz2_db2
                self.update_params_sgd(*gradients)

            print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
            accuracy = self.test_accuracy(test_data, test_labels)
            print(f'Accuracy: {accuracy:.2f}')

# Load and preprocess the data
# https://numpy.org/doc/1.19/reference/generated/numpy.loadtxt.html

train_data_flattened = np.loadtxt('dataset/train/train_images.csv', delimiter=',')  # (209,12288)
train_labels = np.loadtxt('dataset/train/train_labels.csv', delimiter=',')  # (209,)

test_data_flattened = np.loadtxt('dataset/test/test_images.csv', delimiter=',')  # (50,12288)
test_labels = np.loadtxt('dataset/test/test_labels.csv', delimiter=',')  # (50,)

#print(train_data_flattened.shape)
#for i in range(0, 12288):
#  print(" ", test_data_flattened[0][i], end="")

#print(test_labels)

# Normalize the data
# Dividing an image by 255 is a common method to normalize pixel values to the range [0, 1] in Python, particularly when working with image data for machine learning models.
# This operation is typically applied to images stored as integers (e.g., uint8) to convert them into floating-point values suitable for processing. For example, a function might perform this normalization as images = images / 255.
# This approach is widely used and is often referred to as "divide_255" normalization.
# The resulting normalized images have pixel intensities scaled proportionally from the original 0-255 range down to 0.0-1.0.
# This normalization is essential for ensuring consistent input to neural networks and other algorithms that expect data within a specific range.

train_data = train_data_flattened / 255.0
test_data = test_data_flattened / 255.0

# print(train_data.shape)
#for i in range(0, 12288):
#  print(" ", test_data[0][i], end="")


# Define the neural network parameters
input_size = train_data.shape[1]
#print(train_data.shape)
print(input_size)

hidden_size = 128
output_size = 1

# Create the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.001)

# Train the network
nn.train(train_data, train_labels, epochs=200, batch_size=10)

# predict
predictions = nn.predict(test_data)
print(predictions)
# Test the model
accuracy = nn.test_accuracy(test_data, test_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

