# Neural Network Implementations

## Overview
This repository contains two implementations of neural networks built from scratch. The first implementation features a simple feedforward neural network, while the second implementation is a dynamic architecture using a Feedforward Neural Network (FNN). Both serve as foundational examples for learning and understanding neural network algorithms.

## Files

### 1. SimpleFNN.py
This script implements a basic feedforward neural network with the following features:
- **Input Layer**: Accepts data with three features.
- **Hidden Layer**: Utilizes the ReLU activation function.
- **Output Layer**: Produces a single output.
- **Training Method**: Uses Stochastic Gradient Descent (SGD) for optimization.

### 2. NeuralNetwork.py
This script provides a more flexible implementation of a neural network with dynamic architecture:
- **Dynamic Layer Sizes**: Allows for customizable layer sizes.
- **Activation Functions**: Implements ReLU and its derivative.
- **Training Methods**: Supports both Stochastic Gradient Descent (SGD) and Full-Batch Gradient Descent (FBGD).
- **Prediction**: Capable of making predictions based on trained models.

## Getting Started

### Prerequisites
Ensure you have Python 3 and NumPy installed:
```bash
pip install numpy