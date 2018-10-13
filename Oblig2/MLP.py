"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import statistics


class mlp:
    def __init__(self, inputs, targets, hidden):
        """
        Initialize the class attributes
        :param inputs: Input data
        :param targets: Target data
        :param hidden: Number of hidden nodes in the layer
        """
        # Initiate hyperparamters
        self.beta = 1
        self.eta = 0.01  # Learning rate
        self.momentum = 0.0  # To push out of local optimum/minimum
        self.bias = -1  # Input for the bias node
        self.inputs = inputs  # Make an attribute of the input data
        self.targets = targets
        self. hidden = hidden
        self.output = 8

        # Initiate weights
        self.weights_input = self.initialize_weights(len(self.inputs[0, :])+1, hidden+1)
        self.weights_output = self.initialize_weights(hidden+1, self.output)

        # Attributes are initiated in methods
        self.error_validation = []
        self.error_training = []
        self.a = None
        self.h = None
        self.h_ = None

    def initialize_weights(self, n: int, output: int):
        """
        Initialize weights. Will add a bias node weights for the output and hidden layer.
        :param output: Number of outputs
        :param n: Number of inputs
        :return:
        """
        # Initiate weights randomly distributed between the low-high interval
        weights = np.random.uniform(low=(-1 / math.sqrt(n)),
                                    high=1 / math.sqrt(n),
                                    size=(output, n))

        return weights

    def k_fold_cross_vaildation(self, inputs, targets, k=8):
        """
        Trains a model on k number of groups  from the dataset.
        splits the data set into k-1 training sets and 1 test set,
        which will evaluate the efficiency of the model.

        :param inputs: Dataset
        :param targets: Labels
        :param valid: Dataset for validation
        :param validtargets: Label for validation set
        :param k: Number of folds
        :return:
        """

        # Set up accuracy and confusion lists for all the folds
        accuracy, confusion = [], []

        # Raises an error if the chosen k does not make an equal set of groups.
        if len(inputs) % k != 0:
            raise ValueError("The k is not equally divided by the length of the dataset. Choose another k.")

        # Split the data into k equally large groups.
        self.data_groups = np.split(inputs, k)
        self.label_groups = np.split(targets, k)

        # Train k times
        for i in range(k):

            # Initiate data attributes. Must be reset each time
            self.training_set = []
            self.training_set_target = []
            self.test_set = []
            self.test_set_target = []
            self.validation_set = []
            self.validation_set_target = []

            # Set up training, validation and test
            for data, labels in zip(self.data_groups[:-2], self.label_groups[:-2]):
                self.training_set.append(data)
                self.training_set_target.append(labels)
            self.training_set = np.concatenate(self.training_set)
            self.training_set_target = np.concatenate(self.training_set_target)
            self.validation_set = self.data_groups[-1]
            self.validation_set_target = self.label_groups[-1]
            self.test_set = self.data_groups[-2]
            self.test_set_target = self.label_groups[-2]

            # Rotate group k times
            self.data_groups = self.rotate(self.data_groups)
            self.label_groups = self.rotate(self.label_groups)

            # Initialize new weights
            self.weights_input = self.initialize_weights(len(self.inputs[0, :]) + 1, self.hidden + 1)
            self.weights_output = self.initialize_weights(self.hidden + 1, self.output)

            # Start training on current set
            self.earlystopping(self.training_set, self.training_set_target,
                               self.validation_set, self.validation_set_target)


            conf, acc = self.confusion(self.test_set, self.test_set_target)
            confusion.append(conf)
            accuracy.append(acc)

        # Print results
        print("The accuracy mean: ", statistics.mean(accuracy))
        print("The accuracy std:  ", statistics.stdev(accuracy))
        print("The best accuracy: ", max(accuracy))

    def rotate(self, list, rotations=1):
        """
        Rotates the group of data
        :param list: Data groups
        :param rotations: Number of rotations
        :return:
        """
        return list[rotations:] + list[:rotations]

    def earlystopping(self, inputs, targets, valid, validtargets, plot = False):
        """
        Early stopping method that check validation error versos training error.
        Will stop the training if the validation error increases compared to the
        training error.
        :param inputs:
        :param targets:
        :param valid:
        :param validtargets:
        :return:
        """
        # Initiate plotting parameters
        epoch_plot = []  # Plot axis
        epoch = 0  # Epoch counter
        overfitting = 0  # Earlystop counter if overfitting

        # Reset every time in case k-fold is used
        self.error_validation = []
        self.error_training = []
        self.a = None
        self.h = None
        self.h_ = None

        # Sets up the axis for live plotting
        if plot:
            plt.plot(epoch_plot, self.error_validation, color='red', label='Validation error')
            plt.plot(epoch_plot, self.error_validation, color='blue', label='Training error')
            plt.legend()

        # Continue training until earlystop is activated
        while True:
            correct = 0
            not_correct = 0
            error = []

            # Perform a validation check.
            for input_vector, output_vector in zip(valid, validtargets):

                # Move forward
                prediction = self.forward(input_vector)
                error.append(self.sum_of_squares_error(output_vector, prediction))

                # Compare prediction to the target
                pred_max_idx = prediction.index(max(prediction))
                target_max_idx = np.argmax(output_vector)
                if pred_max_idx == target_max_idx:
                    correct += 1
                else:
                    not_correct += 1

            # Train weights
            self.train(inputs, targets)

            # Take the mean of the error from the validation
            self.error_validation.append(statistics.mean(error))

            # Check if there is time to stop.
            if epoch > 10:

                # Check current error against the average of the last 20 or if converging
                if self.error_validation[-1] > statistics.mean(self.error_validation[-20:]):
                    overfitting += 1

                # Time to stop if if was worse 10 times or same 100 times.
                if overfitting == 50:
                    print("Earlystop activated. Evaluating testset...")
                    break

            epoch_plot.append(epoch)
            epoch += 1

            # Plot the error rates
            if plot:
                plt.plot(epoch_plot, self.error_validation, color='red', label='Validation error')
                plt.plot(epoch_plot, self.error_training, color='blue', label='Training error')
                plt.pause(1/10**10)

            # Print results
            if epoch % 10 == 0:
                result = correct / (not_correct + correct)
                print("epoch {}: Error: {}, Accuracy: {}"
                      .format(epoch, self.error_validation[-1], result))

        if plot:
            # Save the figure
            plt.savefig("{}nodes.png".format(self.hidden))

    def train(self, inputs, targets):
        """
        Trains the network with a backproporgation algorithm. First goes forward, then
        trains by calculating the error backwards.
        :param inputs:
        :param targets:
        :param iterations:
        :return:
        """

        # Shuffle the order of the input and target
        combined = list(zip(inputs, targets))
        random.shuffle(combined)
        a, b = zip(*combined)

        error = []

        # Insert input
        for input_vector, output_vector in zip(a, b):
            # # Add bias to the input
            # input_vector = np.insert(input_vector, 0, self.bias)

            # Go forward through the net and predict an output
            prediction = self.forward(input_vector)

            # Compute error
            error.append(self.sum_of_squares_error(output_vector, prediction))

            # Go backwards, compute error and update the weights.
            self.backwards(output_vector, input_vector, prediction)

        self.error_training.append(statistics.mean(error))

    def backwards(self, output_vector, input_vector, pred):
        """
        Computes the error backwards through the net
        :param output_vector: The target
        :param pred: The predicted values
        :return:
        """
        # Add bias to the input
        input_vector = np.insert(input_vector, 0, self.bias)

        # Compute the error at the output
        self.delta_output = self.compute_delta_output(output_vector, pred)

        # Compute the error in the hidden layer
        self.delta_hidden = self.compute_delta_hidden(self.a, self.weights_output, self.delta_output)

        # Update weights in the output from the error. Don't change the bias for now
        self.weights_output = self.update_weights(self.a, self.weights_output, self.delta_output)
        self.weights_input = self.update_weights(input_vector, self. weights_input, self.delta_hidden)

    def update_weights(self, activation, weights, delta):
        """
        Update the weights for a layer using the input from the previous nodes.
        :param activation:
        :param weights:
        :param delta_output:
        :return:
        """
        for k in range(len(weights[0])):
            for j in range(len(weights)):
                weights[j][k] = weights[j][k] - self.eta * delta[j] * activation[k]
        return weights

    def forward(self, input_vector):
        """
        Takes a input and moves it forward through the net
        :param input_vector:
        :return: A prediction. Uses linear output.
        """
        # Add bias to the input
        input_vector = np.insert(input_vector, 0, self.bias)

        # Go through hidden layer
        self.h = self.weighted_sum(input_vector, self.weights_input, self.hidden)
        self.a = self.sigmoid_activation(self.h)

        # From hidden layer to output.
        h_ = self.weighted_sum(self.a, self.weights_output, self.output)

        return h_[:-1]

    def sigmoid_activation(self, h):
        """
        Sigmoid activation functions
        :param h: Weighted sum
        :return: Activation result
        """
        return [1/(1+math.exp(-self.beta*node)) for node in h]

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

        delta_hidden = []
        # Loop through each node and calculate the delta error. Don't include the bias.
        for n in range(len(activation)):
            # Calculate the sum of the weights and output error.
            w_sum = sum([weights[i][n]*delta_output[i] for i in range(len(delta_output))])

            # Multiply with the differential of the activation function
            delta_hidden.append(activation[n] * (1 - activation[n]) * w_sum)

        return delta_hidden

    def sum_of_squares_error(self, target, y):
        """
        Calculates the squared sum of the error.

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
            w_sum = sum([weights[n][i]*input[i] for i in range(len(input))])
            h.append(w_sum)
        h.append(self.bias)

        # Return the weighted sum for the neurons.
        return h

    def confusion(self, inputs, targets):
        """
        Runs through a test set and creates a confusion matrix
        :param inputs: Test set
        :param targets: Labels
        :return: confusion matrix and accuracy
        """

        # initiate matrix
        confusion = np.zeros((8, 8), dtype=int)

        # Loop through test set
        for input_vector, output_vector in zip(inputs, targets):
            # Make a prediction
            prediction = self.forward(input_vector)

            # Get indexes to the confusion matrix
            prediction_index = prediction.index(max(prediction))
            target_index = np.argmax(output_vector)

            # Add results to confusion matrix

            confusion[target_index][prediction_index] += 1

        # Print the accuracy and confusion matrix
        print("\nConfusion matrix:")
        print(confusion)
        result = np.trace(confusion)/np.sum(confusion)
        print("Test accuracy: {}%".format(round(result*100, 3)))
        return confusion, result


