#!/usr/bin/env python3
"""
This module contains the driver code for the exercise part
"""

import mnist_loader
import network

# Load the MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create a network with 784 input neurons, 30 hidden neurons, and 10 output neurons
net = network.Network([784, 10])

# Train the network using stochastic gradient descent
# 30 epochs, mini-batch size of 10, and learning rate of 3.0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
