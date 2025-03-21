import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):

        np.random.seed (42) 
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)

        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            self.weights.append((np.random.randn(layer_sizes[i], layer_sizes[i+1])) * np.sqrt(1. / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    
    def forward(self, X): # return the output of the last layer
        self.activations = [X] # input of the each layer 
        self.z_store = []
        for i in range(self.num_layers - 1):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_store.append(z)

            if i == self.num_layers - 2: # last layer
                self.activations.append(z)
            else: # hidden layer
                self.activations.append(self.relu(z))
        
        return self.activations[-1]


    def backward(self, y_true, y_pred):

        loss_derivative = y_pred - y_true    
        
        grad_weights = [None] * (self.num_layers - 1)
        grad_biases = [None] * (self.num_layers - 1)

        # Calculate the gradient of the weights and biases from backward
        dL_dz = loss_derivative 

        for i in reversed(range(self.num_layers - 1)):
            grad_weights[i] = np.dot(self.activations[i].T, dL_dz)
            grad_biases[i] = np.sum(dL_dz, axis=0, keepdims=True)

            if i > 0:
                dL_da = np.dot(dL_dz, self.weights[i].T)
                dL_dz = dL_da * self.relu_derivative(self.z_store[i-1])
            
        return grad_weights, grad_biases

    def update_parameter(self, grad_weights, grad_biases):
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]

    def train(self, X_train, y_train, method, epochs=100):
        if method == "SGD":
            for epoch in range(epochs):
                total_loss = 0 
                for i in range(len(X_train)):  # Stochastic Gradient Descent (SGD)
                    X = X_train[i].reshape(1, -1)  # (1, 3)
                    y = y_train[i].reshape(1, -1)  # (1, 1)
                
                    y_pred = self.forward(X)  # Forward pass
                    loss = 0.5 * np.square(y - y_pred)  # MSE Loss
                    total_loss += loss
                
                    grad_w, grad_b = self.backward(y, y_pred)  # Backward pass
                    self.update_parameter(grad_w, grad_b)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, Loss: {total_loss[0][0]:.4f}")

        elif method == "FBGD":
            for epoch in range(epochs):
                y_pred = self.forward(X_train)  
                total_loss = 0.5 * np.mean(np.square(y_train - y_pred))  
                grad_w, grad_b = self.backward(y_train, y_pred)  
                self.update_parameter(grad_w, grad_b)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    
    def predict(self, X_test):
        return self.forward(X_test)


X_train = np.array([[2, 3, 5],
                    [4, 2, 1],
                    [6, 5, 2]])

y_train = np.array([[12], [8], [15]])

X_test = np.array([[3, 4, 3]])  

# Build up the neural network model
nn = NeuralNetwork(layer_sizes=[3, 3, 3, 1], learning_rate=0.01)

# training the model
nn.train(X_train, y_train, method="SGD", epochs=200)

# Prediction
y_pred = nn.predict(X_test)
print(f"\nTest Data: {X_test.tolist()}")
print(f"Predicted Output: {y_pred[0][0]:.4f}")