"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt


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
        self.weights_output = self.initialize_weights(self.output, hidden, self.weight_init)  # Number of outputs


    def initialize_weights(self, n: int, output: int, weight_init=None):
        """
        Initialize weights. Will add a bias node weights for the output and hidden layer.
        :param output: Number of outputs
        :param n: Number of inputs
        :param weight_init: Can be used as argument ot initialize new sets of weight with about the same values.
        :return:
        """


        if weight_init == None:
            # Random initialization of weights between -1/sqrt(n) < w < 1/sqrt(n)
            self.weight_init = random.sample([-1 / math.sqrt(n), 1/math.sqrt(n)], 1)

        # Make a list of the weights for each node. +1 is to make a bias node.
        w_input = []
        for node in range(output+1):
            w_input.append([self.weight_init[0] + random.random() * 0.01 for i in range(n)])

        # Return list of weights with about the same size, but with a small random variance.
        return w_input

    def earlystopping(self, inputs, targets, valid, validtargets):
        print('To be implemented')

    def train(self, inputs, targets, iterations=100):
        """
        Trains the network with a backproporgation algorithm. First goes forward, then
        trains by calculating the error backwards.
        :param inputs:
        :param targets:
        :param iterations:
        :return:
        """
        for epoch in range(100):
            correct = 0
            not_correct = 0
            # Initiate a training input
            for input_vector, output_vector in zip(inputs, targets):

                # If sequential update, shuffle the order of the input vector.
                # random.shuffle(input_vector)

                # Go forward through the net and predict an output
                pred = self.forward(input_vector)

                # Compare prediction to the target
                pred_max_idx = pred.index(max(pred))
                target_max_idx = np.argmax(output_vector)
                if pred_max_idx == target_max_idx:
                    correct += 1
                else:
                    not_correct += 1

                # Go backwards, compute error and update the weights.
                self.backwards(output_vector, input_vector, pred)

            print(correct, not_correct)



        # plt.show()

    def backwards(self, output_vector, input_vector, pred):
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

        # Update weights in the output from the error. Don't change the bias for now

        # TODO self.h_ should be self.a and self.a should inputvector. Or something like that
        self.weights_output = self.update_weights(self.a, self.weights_output, self.delta_output)
        self.weights_input = self.update_weights(input_vector, self. weights_input, self.delta_hidden)

    def update_weights(self, activation, weights, delta):
        """
        Update the weights for a layer.
        :param activation:
        :param weights:
        :param delta_output:
        :return:
        """
        # Don't include bias for now. Try with self.h_, but it might be that I should use
        # the activation for the hidden layer. But is the size of 12 not 8.
        updated_weights = []
        # Loop through each node
        for i, weight in enumerate(weights[:-1]):
            w_up = []

            # Update the weights.
            for d, w in zip(delta, weight):
                w_up.append(w - self.eta*activation[i]*d)

            # Append the update for each node
            updated_weights.append(w_up)

        # Update the bias
        w_up = []
        for d, w in zip(delta, weights[-1]):
            w_up.append(w - self.eta * self.bias * d)
        updated_weights.append(w_up)

        return updated_weights

    def forward(self, inputs):
        """
        Forward phase through the net.
        :param inputs: Input vector
        :return:
        """
        # Go through hidden layer
        self.h = self.weighted_sum(inputs, self.weights_input, self.hidden)
        self.a = self.sigmoid_activation(self.h)

        # From hidden layer to output.
        self.h_ = self.weighted_sum(self.a, self.weights_output, self.output)

        # h_ is the linear activation output
        return self.h_

    def sigmoid_activation(self, h):
        """
        Sigmoid activation functions
        :param h: Weighted sum
        :return: Activation result
        """
        try:
            return [1/(1+math.exp(-self.beta*node)) for node in h]
        except OverflowError:
            print("stop")

    def compute_delta_output(self, target, y):
        """
        Computes the delta error for the output. Uses the linear output error
        Takes the squared error
        :param target: Target output
        :param y: Predicted output
        :param a: activation output
        :return:
        """
        # Compute the delta output error.
        error = [(y_k - t_k) for t_k, y_k in zip(target, y)]
        return error

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
        # Loop through each node and calculate the delta error. Don't include the bias.
        for n, weight in enumerate(weights[:-1]):

            # Calculate the sum of the weights and output error.
            w_sum = sum([(weight*output) for weight, output in zip(weight, delta_output)])

            # Multiply with the differential of the activation function
            delta_hidden.append(activation[n]*(1-activation[n])*w_sum)

        return delta_hidden

    def sum_of_squares_error(self, target, y):
        """
        Sum of Squared Error function
        Takes the squared error
        :param target:
        :param y:
        :return:
        """
        return 1/2*sum([(target - y)**2 for target, y in zip(target, y)])

    def weighted_sum(self, input, weights, output):
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
            w_sum = [(x*w) for x, w in zip(input, weights[n])]
            w_sum.append(-1*weights[-1][n])  # Add the bias
            h.append(sum(w_sum))  # Sum it

        # Return the weighted sum for the neurons.
        return h

    def confusion(self, inputs, targets):
        print('To be implemented')


