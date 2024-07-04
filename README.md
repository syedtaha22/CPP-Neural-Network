# Neural Network Theory and Implementation in C++

## Abstract

This document presents a comprehensive overview of the design, theory, and implementation of a neural network. The aim is to provide a detailed exposition of the neural network architecture, including forward and backward propagation mechanisms, loss calculation, and training processes. This document integrates theoretical concepts with practical implementation using C++.

## Table of Contents

1. [Introduction](#introduction)
2. [Neural Network Architecture](#neural-network-architecture)
    - [Input Layer](#input-layer)
    - [Hidden Layers](#hidden-layers)
        - [First Hidden Layer](#first-hidden-layer)
        - [Second Hidden Layer](#second-hidden-layer)
    - [Output Layer](#output-layer)
3. [Activation Functions](#activation-functions)
    - [ReLU Activation Function](#relu-activation-function)
    - [Linear Activation Function](#linear-activation-function)
4. [Loss Calculation](#loss-calculation)
    - [Mean Squared Error (MSE)](#mean-squared-error-mse)
5. [Backpropagation](#backpropagation)
    - [Gradient Computation](#gradient-computation)
    - [Weight and Bias Updates](#weight-and-bias-updates)
6. [Training Process](#training-process)
7. [Model Evaluation](#model-evaluation)
8. [Implementation](#implementation)
    - [Code Overview](#code-overview)
    - [Data Generation](#data-generation)
    - [Training and Evaluation](#training-and-evaluation)
9. [Conclusion](#conclusion)

---

## Introduction

Neural networks are powerful tools for machine learning, capable of modeling complex relationships and making predictions based on data. This document elucidates the design and implementation of a feedforward neural network with two hidden layers, focusing on theoretical foundations and practical coding aspects in C++.

## Neural Network Architecture

### Input Layer

- **Description**: The input layer receives the raw data features for processing.
- **Shape**: Single input feature.

### Hidden Layers

#### First Hidden Layer

- **Shape**:
  - **Inputs**: 1
  - **Units**: 10
  - **Weights**: Randomly initialized (Shape: 1x10)
  - **Biases**: Randomly initialized (Shape: 10)
- **Feed Forward**:
  - **Weighted Sum**: $Z_1 = X \cdot w_1 + b_1$
  - **Activation Function**: $A_1 = ReLU(Z_1)$

#### Second Hidden Layer

- **Shape**:
  - **Inputs**: 10
  - **Units**: 10
  - **Weights**: Randomly initialized (Shape: 10x10)
  - **Biases**: Randomly initialized (Shape: 10)
- **Feed Forward**:
  - **Weighted Sum**: $Z_2 = A_1 \cdot w_2 + b_2$
  - **Activation Function**: $A_2 = ReLU(Z_2)$

### Output Layer

- **Description**: Produces the final output of the network.
- **Shape**:
  - **Inputs**: 10
  - **Units**: 1
  - **Weights**: Randomly initialized (Shape: 10x1)
  - **Biases**: Randomly initialized (Shape: 1)
- **Feed Forward**:
  - **Weighted Sum**: $Z_{\text{output}} = A_2 \cdot w_{\text{output}} + b_{\text{output}}$
  - **Activation Function**: Linear function $Output = Z_{\text{output}}$

## Activation Functions

### ReLU Activation Function

- **Function**: $ReLU(x) = max(0, x)$
- **Derivative**: Returns 1 if $x > 0$, else 0.

### Linear Activation Function

- **Function**: $Linear(x) = x$
- **Derivative**: Always returns 1.

## Loss Calculation

### Mean Squared Error (MSE)

- **Definition**: Measures the average squared difference between predicted values and actual values.
- **Formula**: $Loss = \frac{1}{2} \cdot (actual - prediction)^2$

## Backpropagation

Backpropagation is a fundamental algorithm in training neural networks, designed to compute the gradient of the loss function with respect to each weight in the network. This process is essential for optimizing the network's parameters through gradient descent. The key idea behind backpropagation is the systematic application of the Chain Rule to propagate the error gradient from the output layer back through each preceding layer.

#### Gradient Computation

1. **Output Layer Gradients**:
   - **Derivative of Loss**: $\frac{dL}{dZ_{\text{output}}} = \text{prediction} - \text{actual}$
   - **Weight Gradient**: $\frac{dL}{dw_{\text{output}}} = A_2^T \cdot \frac{dL}{dZ_{\text{output}}}$
   - **Bias Gradient**: $\frac{dL}{db_{\text{output}}} = \frac{dL}{dZ_{\text{output}}}$

2. **Hidden Layer 2 Gradients**:
   - **Derivative of Loss**: $\frac{dL}{dZ_2} = \frac{dL}{dZ_{\text{output}}} \cdot w_{\text{output}}^T \cdot \text{ReLU'}(Z_2)$
   - **Weight Gradient**: $\frac{dL}{dw_2} = A_1^T \cdot \frac{dL}{dZ_2}$
   - **Bias Gradient**: $\frac{dL}{db_2} = \frac{dL}{dZ_2}$

3. **Hidden Layer 1 Gradients**:
   - **Derivative of Loss**: $\frac{dL}{dZ_1} = \frac{dL}{dZ_2} \cdot w_2^T \cdot \text{ReLU'}(Z_1)$
   - **Weight Gradient**: $\frac{dL}{dw_1} = X^T \cdot \frac{dL}{dZ_1}$
   - **Bias Gradient**: $\frac{dL}{db_1} = \frac{dL}{dZ_1}$
### Weight and Bias Updates

#### Weight and Bias Updates

- **Gradient Descent Update**:
  - $w_{\text{output}} = w_{\text{output}} - learning\\\_rate \cdot \frac{dL}{dw_{\text{output}}}$
  - $b_{\text{output}} = b_{\text{output}} - learning\\\_rate \cdot \frac{dL}{db_{\text{output}}}$
  - $w_2 = w_2 - learning\\_rate \cdot \frac{dL}{dw_2}$
  - $b_2 = b_2 - learning\\_rate \cdot \frac{dL}{db_2}$
  - $w_1 = w_1 - learning\\_rate \cdot \frac{dL}{dw_1}$
  - $b\_1 = b_1 - learning\\_rate \cdot \frac{dL}{db_1}$

## Training Process

1. **Initialize Network**: Set up the architecture with initial weights and biases.
2. **Forward Pass**: Compute predictions by propagating inputs through the network.
3. **Loss Calculation**: Compute the loss using predicted and actual values.
4. **Backpropagation**: Update weights and biases to minimize the loss.
5. **Iteration**: Repeat the process for a predefined number of epochs or until convergence.

## Model Evaluation

- **Accuracy Measurement**: Evaluate the trained network on test data to assess performance using metrics like accuracy, precision, and recall.

## Implementation

### Code Overview

The provided C++ code implements a feedforward neural network with two hidden layers and an output layer. It includes:

- **Activation Functions:** ReLU and Linear.
- **Dense Layer Class:** Manages weights, biases, forward, and backward passes.
- **Model Class:** Coordinates layers, performs training, prediction, and evaluation.
- **DataGenerator Class:** Functor class for generating data points.
- **Main Function:** Generates data, trains the model, and saves predictions.

### Data Generation

- **Training Data**: Generated for input values ranging from -6.0 to 6.0 with corresponding sine values.
- **Testing Data**: Generated for the same range with a lower resolution.

### Training and Evaluation

- **Training**: The model is trained over 10,000 epochs with a learning rate of 0.01.
- **Evaluation**: Predictions are saved to a file for each epoch to track model performance. This file can then be parsed to get data for each epoch and plotted with matplotlib in Python.

## Conclusion

This document provides a detailed theoretical and practical guide to neural network design and implementation. The integration of theoretical concepts with practical coding in C++ offers a comprehensive understanding of neural network operations and training mechanisms. Neural networks play a pivotal role in modern machine learning, and understanding their implementation helps in harnessing their full potential for various applications.