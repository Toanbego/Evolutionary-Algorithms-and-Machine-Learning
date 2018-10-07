"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np
import math
import random



class mlp:
    def __init__(self, inputs, targets, hidden):
        """
        Intialize hyper parameters
        :param inputs:
        :param targets:
        :param hidden:
        """
        self.beta = 1
        self.eta = 0.1  # Learning rate
        self.momentum = 0.0  # To push out of local optimum/minimum
        self.bias = -1  # Input for the bias node
        self.inputs = inputs  # Make an attribute of the input data
        self.targets = targets
        self. hidden = hidden
        self.output = 8
        self.Error = 0
        self.weights_input = self.initialize_weights(len(self.inputs[0, :]), hidden)
        self.weights_output = self.initialize_weights(self.output, hidden)  # Number of outputs


    def initialize_weights(self, n: int, output: int):
        """
        Initialize weights. Will add a bias node weights for the output and hidden layer.
        :param output: Number of outputs
        :param n: Number of inputs
        :return:
        """

        # Random initialization of weights between -1/sqrt(n) < w < 1/sqrt(n)
        weight_init = random.sample([-1 / math.sqrt(n), 1/math.sqrt(n)], 1)

        # Make a list of the weights for each node. +1 is to make a bias node.
        w_input = []
        for node in range(output+1):
            w_input.append([weight_init[0] + random.random() * 0.09 for i in range(n)])

        # Return list of weights with about the same size, but with a small random variance.
        return w_input

    def earlystopping(self, inputs, targets, valid, validtargets):
        print('To be implemented')

    def train(self, inputs, targets, iterations=100):
        """
        Trains the network with a backproporgation algorithm
        :param inputs:
        :param targets:
        :param iterations:
        :return:
        """

        # Initiate a training input
        for input_vector, output_vector in zip(inputs, targets):

            # If sequential update, shuffle the order of the input vector.
            random.shuffle(input_vector)

            # Go forward through the net and predict an output
            pred = self.forward(input_vector)

            # Calculate the error using target and prediction
            # This error is a scalar, and is the total error
            self.error = self.sum_of_squares_error(output_vector, pred)

            # Go backwards and comput the errors
            self.backwards(output_vector, pred)

    def backwards(self, output_vector, pred):
        """
        Computes the error backwards through the net
        :param output_vector: The target
        :param pred: The predicted values
        :return:
        """
        # Compute the error at the output
        self.delta_output = self.compute_delta_output(output_vector, pred)

        # Compute the error in the hidden layer
        self.delta_hidden = self.compute_delta_hidden(self.a, self.weights_output, self.delta_output)

        # Update weights in the output from the error
        self.weights_output = self.update_weights(self.h_, self.weights_output, self.delta_output)

    def update_weights(self, activation, weights, delta):
        """
        Update the weights for a layer
        :param activation:
        :param weights:
        :param delta_output:
        :return:
        """
        # Don't include bias for now. Try with self.h_, but it might be that I should use
        # the activation for the hidden layer. But is the size of 12 not 8.
        for n, node in enumerate(weights[:-1]):



    def compute_delta_hidden(self, activation, weights, delta_output):
        """
        Computes the delta error for the hidden layer
        :param activation:
        :param weights:
        :param delta_output:
        :return:
        """
        # TODO Maybe calculate the error for the bias node, but for now don't include it.
        delta_hidden = []
        # Loop through each node and calculate the delta error
        for n, node in enumerate(weights[:-1]):

            # Calculate the sum of the weights and output error. Don't include the bias.
            w_sum = sum([(weight*output) for weight, output in zip(node, delta_output)])

            # Multiply with the differential of the activation function
            delta_hidden.append(activation[n]*(1-activation[n])*w_sum)

        return delta_hidden

    def forward(self, inputs):
        """
        Forward phase through the net.
        :param inputs: Input vector
        :return:
        """
        # Go through hidden layer
        self.h = self.weighted_sum(inputs, self.weights_input, self.hidden)
        self.a = self.sigmoid(self.h)

        # From hidden layer to output
        self.h_ = self.weighted_sum(self.a, self.weights_output, self.output)

        # Find the largest number and make a boolean matrix
        # output_idx = h_.index(max(h_))
        # output = [0]*len(h_)
        # output[output_idx] = 1
        return self.h_

    def sigmoid(self, h):
        """
        Sigmoid activation functions
        :param h: Weighted sum
        :return: Activation result
        """
        return [1/(1+math.exp(-self.beta*node)) for node in h]

    def compute_delta_output(self, target, y):
        """
        Computes the delta error for the output
        Takes the squared error
        :param target: Target output
        :param y: Predicted output
        :param a: activation output
        :return:
        """
        # Compute the delta output error.
        error = [(y_k - t_k) * y_k * (1 - y_k) for t_k, y_k in zip(target, y)]
        return error

    def sum_of_squares_error(self, target, y):
        """
        Sum of Squared Error function
        Takes the squared error
        :param target:
        :param y:
        :return:
        """
        return 1/2*sum([(target - y)**2 for target, y in zip(target, y)])

    def weighted_sum(self, input, weight, output):
        """
        Calculates the weighted sum for a node
        :param input: list - the input vector
        :param weight: list - The weights
        :param output: int - Number of outputs
        :return:
        """
        h = []
        # Loop through nodes and calculate weighted sum
        for n in range(output):

            # Loop through input and corresponding weight
            w_sum = [(x*w) for x, w in zip(input, weight[n])]
            w_sum.append(weight[-1][n])  # Add the  bias
            h.append(sum(w_sum))  # Sum it

        # Return the weighted sum for the neurons.
        return h

    def confusion(self, inputs, targets):
        print('To be implemented')


