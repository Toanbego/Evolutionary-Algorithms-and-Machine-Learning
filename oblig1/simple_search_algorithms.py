"""
Author - Torstein Gombos
Created - 06.09.2018

Module with some simple search algorithms that can be used for optimization.
These algorithms are generalized to fit more than one problem.
"""
import random

from oblig1 import routes as r

def gradient_search(f_derv, start_point=3, descent_or_ascent="ascent"):

    """
    Function performs gradient ascent/descent for
    given function derivative with a start point.
    Returns the local maximum
    :param f_derv: derivative of function
    :param start_point: Starting point for search
    :param descent_or_ascent: Find local min or max
    :return: local min/max
    """

    # Initialize parameters
    cur_x = start_point  # The algorithm starts at x=3
    rate = 0.01  # Learning rate
    precision = 0.000001  # This tells us when to stop the algorithm
    previous_step_size = 1  #
    max_iters = 10000  # maximum number of iterations
    iters = 0  # iteration counter

    # Perform gradient search
    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x  # Store current x value in prev_x
        if descent_or_ascent == "descent":
            cur_x = cur_x - rate * f_derv(prev_x)  # Grad descent
        if descent_or_ascent == "ascent":
            cur_x = cur_x + rate * f_derv(prev_x)  # Grad descent
        previous_step_size = abs(cur_x - prev_x)  # Change in x
        iters = iters + 1  # iteration count


    return cur_x

def one_swap_crossover(ind):
    """
    Swaps two random alleles with each other
    :param ind: The individual to perform crossover
    :return: Mutated individual
    """
    seq_idx = range(len(ind))
    # Sample two random alleles and swap them
    a1, a2 = random.sample(seq_idx, 2)
    ind[a1], ind[a2] = ind[a2], ind[a1]
    return ind

def hill_climber(data):
    """
    Hill climber algorithm that will check a neighboring solution
    If the neighbor solution is better, this becomes the new solution
    if not, keep the old one.
    :param data:
    :param search_space:
    :return:
    """
    # Set up a route with 24 cities
    route = r.create_random_route()
    precision = 0.01
    one_swap_crossover(route)
    # solution =
    #
    while solution > precision:
        r.get_total_distance(data, solution)

def exhaustive_search(f, data, search_space):
    """
    Function that searches every possible solution and returns global minimum
    :param f: Function
    :return: Returns y and x value
    """
    step = 0.001
    # Arbitrary start value
    max_value = f(data, search_space[0])
    for step in search_space:
        new_value = f(data, step)
        if new_value < max_value:
            max_value = new_value
            x_value = step
    return max_value, x_value

