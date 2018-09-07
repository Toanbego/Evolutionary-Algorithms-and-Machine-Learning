"""
Author - Torstein Gombos
Created - 06.09.2018

Module with some simple search algorithms that can be used for optimization.
"""
import sys, os
sys.path.append('C:/Users/toanb/Documents/Skole_programmering/INF4490/oblig1')
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
    # Sample two random alleles and swap them
    seq_idx = list(range(len(ind)))
    a1, a2 = random.sample(seq_idx, 2)
    # Do not swap start and end location. If so, re-sample.
    while a1 == seq_idx[0] or a1 == seq_idx[-1] or a2 == seq_idx[0] or a2 == seq_idx[-1]:
        a1, a2 = random.sample(seq_idx, 2)
    ind[a1], ind[a2] = ind[a2], ind[a1]
    return ind


def hill_climber(data, route_length=24, num_of_rand_resets=10000):
    """
    Hill climber algorithm that will check a neighboring solution
    If the neighbor solution is better, this becomes the new solution
    if not, keep the old one.
    :param data:
    :param max_searches
    :return:
    """

    times_improved = 0
    for explorations in range(num_of_rand_resets):
        # Set up random route
        route = r.create_random_route(route_length)  # Set up a route with 24 cities
        travel_distance = r.get_total_distance(data, route)  # Initiate start solution
        searches = 1
        max_searches = 100
        # Begin climbing
        while searches < max_searches:
            move = False
            # Start moving
            neighbor_route = one_swap_crossover(route)
            # Check that we have not reached end of search space
            if searches >= max_searches:
                break
            updated_distance = r.get_total_distance(data, neighbor_route)
            searches += 1
            if updated_distance > travel_distance:
                route = neighbor_route
                travel_distance = updated_distance
                move = True  # A move was made and it lives to see another day

            if not move:
                break  # No move was made

        # return searches, fitness, route
        # Create a start value
        if explorations == 0:
            fitness = travel_distance
            print("Started with {}".format(fitness))
        if travel_distance < fitness:
            times_improved += 1
            fitness = travel_distance
            best_route = route
    print("Performed 100000 random searches and found a better solution {} times\n".format(times_improved))
    return fitness, best_route

def exhaustive_search(f, data):
    """
    Function that searches every possible solution and returns global minimum
    :param f: Function
    :param data: The data that is needed for some functions
    :param search_space: Possible solutions
    :return: Returns y and x value
    """
    routes = r.create_permutation_of_routes(data, route_length=6)
    fitness = f(data, routes[0])  # Arbitrary start value
    # Loop through all possible solutions and pick the best one
    for step in routes:
        new_value = f(data, step)
        if new_value < fitness:
            fitness = new_value
            x_value = step
    return fitness, x_value
