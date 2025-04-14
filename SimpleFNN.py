import numpy as np

# Step 1 : Define the training and testing data
X_train = np.array([[2, 3, 5],
                    [4, 2, 1],
                    [6, 5, 2]])  # 輸入數據 (3 samples, 3 features)

y_train = np.array([[12], [8], [15]])  # 目標輸出 (3 samples, 1 output)

X_test = np.array([[3, 4, 3]])  # 測試數據

# Step 2 : initialize the weights and biases
W1 = np.random.randn(3,3) * 0.01 # 3 x 3
b1 = np.zeros((1,3)) # 1 x 3

W2 = np.random.randn(3,1) * 0.01 # 3 x 1
b2 = np.zeros((1,1)) # 1 x 1

learning_rate = 0.01
epochs = 100
counter = 0

# Step 3 : Define the activation function
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Step 4 :Training
for epoch in range(epochs):
    
    for i in range(len(X_train)):
        x = X_train[i].reshape(1, 3) # 1 x 3 row vector
        y_true = y_train[i].reshape(1, 1) # 1 x 1

        # Forward pass
        z = np.dot(x, W1) + b1 # 1 x 3
        a = relu(z) # 1 x 3 -> [ReLU(z1), ReLU(z2), ReLU(z)]
    
        z_output = np.dot(a, W2) + b2 # 1 x 1

        # loss function 
        loss = 0.5 * np.square(y_true - z_output)


        # Backward pass
        loss_derivative = z_output - y_true # 1 x 1

        # output layer
        dL_db2 = loss_derivative
        dL_dW2 = loss_derivative * a.T

        # hidden layer
        dL_da = loss_derivative * W2.T # 1 x 3
        dL_db1 = relu_derivative(z) * dL_da # 1 x 3

        dL_dW1 = np.dot(x.T, dL_db1) # 3 x 3 

        # Update weights and biases
        W1 -= learning_rate * dL_dW1
        b1 -= learning_rate * dL_db1
        W2 -= learning_rate * dL_dW2
        b2 -= learning_rate * dL_db2

    counter += 1   
    if counter % 10 == 0:
        print(f"Epoch {counter}, Loss: {loss[0][0]:.4f}")

# Step 5 : Prediction 
Z1_test = np.dot(X_test, W1) + b1
H_test = relu(Z1_test)
y_test_pred = np.dot(H_test, W2) + b2

print(f"\nTest Data: {X_test.tolist()}")
print(f"Predicted Output: {y_test_pred[0][0]:.4f}")
        
    
    

    
    