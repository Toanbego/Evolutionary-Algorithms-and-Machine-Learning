"""
Author - Torstein Gombos
Created - 06.09.2018

Module with some simple search algorithms that can be used for optimization.
"""
import sys, os
sys.path.append('C:/Users/toanb/Documents/Skole_programmering/INF4490/oblig1')
import random
from oblig1 import routes as r

def one_swap_crossover(route):
    """
    Swaps two random alleles with each other
    :param ind: The individual to perform crossover
    :return: Mutated individual
    """
    # Sample two random alleles and swap them
    for swap in range(1000):
        seq_idx = list(range(len(route)))
        a1, a2 = random.sample(seq_idx[1:-1], 2)
        copy = route[:]
        copy[a1], copy[a2] = copy[a2], copy[a1]
        yield copy

def one_swap_crossover_system(route):
    """
    Generates a sequence of random swaps
    :param ind: The individual to perform crossover
    :return: Mutated individual
    """
    # Create a random index swap
    ind1 = list(range(1, len(route)-1))
    ind2 = list(range(1, len(route)-1))
    random.shuffle(ind1)
    random.shuffle(ind2)
    # Loop through cities and swap
    for city1 in ind1:
        for city2 in ind2:
            copy = route[:]
            c1 = copy[city1]
            c2 = copy[city2]
            copy[city1], copy[city2] = copy[city2], copy[city1]
            assert copy[0] == copy[-1], "start and home is not the same"
            yield copy

def hill_climber(data, route_length=24, num_of_rand_resets=100):
    """
    Hill climber algorithm that will check a neighboring solution
    If the neighbor solution is better, this becomes the new solution
    if not, keep the old one.
    :param data:
    :param max_searches
    :return:
    """
    # Set up random route
    route = r.create_random_route(route_length)  # Set up a route with 24 cities
    travel_distance = r.get_total_distance(data, route)  # Initiate start solution
    # Begin climbing, try out a number of swaps
    num_evaluations = 1
    while num_evaluations < 10000:
        move_made = False
        for next_route in one_swap_crossover_system(route):
            if num_evaluations >= 10000:
                break
            updated_dist = r.get_total_distance(data, next_route)
            num_evaluations += 1
            if updated_dist < travel_distance:
                route = next_route
                travel_distance = updated_dist
                move_made = True
                break

        if not move_made:
            break
    print(num_evaluations)
    return travel_distance, route

def exhaustive_search(route_distance, data, route_length=6):
    """
    Function that searches every possible solution and returns global minimum
    :param route_distance: Function
    :param data: The data that is needed for some functions
    :return: Returns y and x value
    """
    # Setup route permutations
    routes = r.create_permutation_of_routes(route_length)
    fitness = route_distance(data, routes[0])  # Arbitrary start value
    # Loop through all possible solutions and pick the best one
    for step in routes:
        new_value = route_distance(data, step)
        if new_value < fitness:
            fitness = new_value
            x_value = step
    return fitness, x_value
