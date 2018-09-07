"""
Author - Torstein Gombos
Created - 06.09.2018

Package with functions regarding routes and distance for routes
"""

import random
import itertools

def get_distance_cities(data, city1, city2) -> float:
    """
    Calculates the distance between two cities from a CSV file
    :param data: The dataset with city names and distance
    :param city1: First city
    :param city2: Second city
    :return: Distance
    """
    idx1, idx2 = data[0].index(city1), data[0].index(city2)
    dist_idx = abs(idx1 - idx2)
    return float(data[min(idx1, idx2)+1][min(idx1, idx2)+dist_idx])

def create_permutation_of_routes(data, route_length=6, random_route=False) -> list:
    """
    Create permutation of a set of cities
    User provides how many cities to be included
    :param route_length: How many cities to be included
    :param random_route: Checks for a random set of cities of the 24 cities
    :return: List of all permutations of the set of cities
    """
    all_routes = []
    if random_route:
        random_route = random.sample(range(24), route_length)
        all_routes = list(itertools.permutations(random_route))
        return all_routes
    else:
        route_starting_point = list(range(route_length))
        # print(route_starting_point)
        # for start_city in route_starting_point.copy():
        #     route_sequence = list(itertools.permutations(route_starting_point))
        #
        #     break
        dist = []
        comb = list(itertools.permutations(route_starting_point))
        for element in comb:
            for i in element:
                dist.append(float(data[1 + element[i]][element[i - 1]]))
        print(dist)
        return all_routes

def create_random_route(route_length = 10):
    """
    Returns a random sequence that can be used to access different indexes
    in the data variable from the CSV file.
    :param route_length: Length of sequence
    :return:
    """
    # Generate a random route sequence
    random.seed()
    random_route = random.sample(range(24), 24)
    return random_route

def create_route(route_length=24):
    return list(range(route_length))

def get_total_distance(data, route, route_length) -> float:
    """
    Sum up the total distance for a route
    :param data: List of data
    :param route: Selected route
    :return: Total route distance
    """
    total_dist = 0
    # loop through route
    new_end_route = route
    for step, travel in enumerate(route):
        try:
            # if step % 120 == 0:
            #     print(new_end_route=new_end_route +2)
            dist = get_distance_cities(data, data[0][travel], data[0][route[step + 1]])

            total_dist += dist
        # Break off when reaching the end of the index
        except IndexError:
            break
    return round(total_dist, 2)
