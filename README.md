# DL_Assignment1

This repository contains an implementation of a Feedforward Neural Network (FFNN) in Python. The network is designed to handle classification tasks and includes various optimization algorithms for training.

## Features

- **Feedforward Neural Network**: The FFNN is implemented with multiple hidden layers and supports different activation functions (sigmoid, ReLU, tanh).
- **Optimization Algorithms**: The code includes several optimization methods:
  - **Gradient Descent (GD)**: Basic gradient descent for updating weights.
  - **Stochastic Gradient Descent (SGD)**: Updates weights using mini-batches.
  - **Momentum Gradient Descent (MGD)**: Incorporates momentum to improve convergence.
  - **Nesterov Accelerated Gradient Descent (NAGD)**: Enhances MGD with Nesterov acceleration.
  - **RMSProp**: Uses the magnitude of recent gradient updates to normalize the gradient.
  - **Adam**: Combines the benefits of RMSProp and MGD.
- **Loss Functions**: Supports cross-entropy loss for classification tasks and mean squared error for regression.
- **Data Preprocessing**: Includes data normalization and one-hot encoding for labels.

## Requirements

- **Python 3.x**
- **NumPy**
- **WandB** (optional, for logging metrics)

## Usage

1. **Initialization**:
   ```python
   from sklearn.model_selection import train_test_split
   import numpy as np
   
   # Example dataset
   x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
   
   # Initialize the FFNN
   ffnn = FFNN(x_train, y_train, x_test, y_test, num_hidden_layers=3, num_neurons=128, act="relu", weight_init="xavier")
   ```

2. **Training**:
   - **Gradient Descent**:
     ```python
     ffnn.gradient_descent(eta=0.01, x=x_train, y_actual=ffnn.y_train_encoded, epochs=10)
     ```
   - **Stochastic Gradient Descent**:
     ```python
     ffnn.sgd(eta=0.01, batch_size=32, epochs=10)
     ```
   - **Momentum Gradient Descent**:
     ```python
     ffnn.mgd(eta=0.01, batch_size=32, epochs=10, beta=0.9)
     ```
   - **Nesterov Accelerated Gradient Descent**:
     ```python
     ffnn.nesterov_agd(eta=0.01, batch_size=32, epochs=10, beta=0.9)
     ```
   - **RMSProp**:
     ```python
     ffnn.rmsprop(eta=0.01, beta=0.9, batch_size=32, epochs=10)
     ```
   - **Adam**:
     ```python
     ffnn.adam(eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=32, epochs=10)
     ```

3. **Evaluation**:
   - **Training Accuracy**:
     ```python
     loss, accuracy, _ = ffnn.cal_accuracy(ffnn.x_train)
     print(f"Training Loss: {loss}, Training Accuracy: {accuracy}")
     ```
   - **Validation Accuracy**:
     ```python
     loss, accuracy, _ = ffnn.cal_accuracy(ffnn.x_val)
     print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
     ```
   - **Test Accuracy**:
     ```python
     loss, accuracy, _ = ffnn.cal_test_accuracy(ffnn.x_test)
     print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
     ```

## Sweep Configuration

To optimize hyperparameters, we used a sweep configuration with the following parameters:

- **Activation Functions**: sigmoid, ReLU, tanh
- **Batch Sizes**: 16, 32, 64
- **Number of Epochs**: 5, 10, 15
- **Number of Hidden Neurons**: 64, 128, 256
- **Learning Rates**: 1e-4, 1e-3, 1e-2
- **Weight Initialization**: Xavier, random

### Sweep Results

The best run was with:
- **Activation Function**: ReLU
- **Batch Size**: 32
- **Number of Epochs**: 10
- **Number of Hidden Neurons**: 128
- **Learning Rate**: 1e-3
- **Weight Initialization**: Xavier

This configuration achieved a **validation accuracy of 89.23%**.


## Transferring to MNIST Dataset

When applying this FFNN to the MNIST dataset, we identified three effective hyperparameter configurations:

1. **4 Hidden Layers, 128 Neurons, Adam Optimizer, Batch Size 32, Learning Rate 1e-3, Xavier Weight Initialization**: Achieved an accuracy of **97.57%**.
2. **3 Hidden Layers, 128 Neurons, Nadam Optimizer, Batch Size 128, Learning Rate 1e-3, Xavier Weight Initialization**: Achieved an accuracy of **97.48%**.
3. **5 Hidden Layers, 128 Neurons, Adam Optimizer, Batch Size 32, Learning Rate 1e-3, Xavier Weight Initialization**: Achieved an accuracy of **97.58%**.

These configurations were selected based on their potential to effectively handle the MNIST dataset.
wandb report:(https://wandb.ai/iitm-ma23m015/DA6401-Assignment_1/reports/MA23M015_DA6401-Assignment-1--VmlldzoxMTQ5ODIyNw)
Github link:(https://github.com/MurugarajT15/DL_Assignment1)
