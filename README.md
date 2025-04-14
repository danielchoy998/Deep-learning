# Simple Neural Network Implementations

This repository contains basic implementations of neural network components using NumPy.

## Convolutional Neural Network (`ConvolutionalNeuralNetwork.py`)

This script provides functions for performing 2D convolutions:

*   `conv2d_with_multi_channel(image, kernel)`: Performs 2D convolution on multi-channel images (e.g., RGB).
*   `conv2d_with_one_channel(image, kernel)`: Performs 2D convolution on single-channel images.
*   `conv2d_with_padding(image, kernel, padding=1)`: Performs 2D convolution with specified padding on single-channel images.

## Feedforward Neural Network (`SimpleFNN.py`)

This script implements a simple Feedforward Neural Network (FNN) with one hidden layer using NumPy from scratch. It covers:
*   Basic data setup (training and testing).
*   Weight and bias initialization.
*   ReLU activation function and its derivative.
*   Forward pass to compute predictions.
*   Mean Squared Error (MSE) loss calculation.
*   Backward pass (backpropagation) to compute gradients.
*   Parameter updates using basic gradient descent.
*   Training loop and prediction on test data.

## Neural Network Class (`NeuralNetwork.py`)

This script defines a more general `NeuralNetwork` class using NumPy:
*   Allows defining networks with multiple layers of arbitrary sizes.
*   Uses ReLU activation for hidden layers (linear for the output layer).
*   Implements Xavier/Glorot initialization for weights.
*   Includes `forward` and `backward` propagation methods.
*   Supports training via:
    *   Stochastic Gradient Descent (SGD) with Momentum.
    *   Full Batch Gradient Descent (FBGD - Note: The implementation seems to update parameters within the loop like SGD, might need review).
*   Provides an `update_parameter` method for weight/bias updates.
*   Includes a `predict` method.

## Self-Attention Mechanism (`Self-attention.py`)

This script implements a basic self-attention layer using PyTorch (`torch.nn.Module`):
*   Takes an embedding size as input.
*   Creates linear layers for Query (Q), Key (K), and Value (V) projections.
*   Calculates scaled dot-product attention scores (Q * K^T / sqrt(d_k)).
*   Applies softmax to the scores to get attention weights.
*   Computes the weighted sum of Values based on the attention weights.
*   Demonstrates usage with random input tensors.

## Work In Progress

*   **Convolutional Neural Network (CNN):** A full implementation from scratch is currently under development.
*   **Self-Attention Mechanism:** An implementation from scratch is planned/in progress. 