"""
Author - Torstein Gombos
Created - 06.09.2018

Package with functions regarding routes and distance for routes
"""
import sys, os
sys.path.append('C:/Users/toanb/Documents/Skole_programmering/INF4490/oblig1')
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

def create_permutation_of_routes(route_length=6, random_route=False) -> list:
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
        for n, element in enumerate(all_routes.copy()):
            for i in element:
                l = list(all_routes[n])
                l.append(i)
                all_routes[n] = tuple(l)
                break
        return all_routes
    else:
        route_starting_point = list(range(route_length))
        all_routes = list(itertools.permutations(route_starting_point[1:]))
        # Appends home destination to all permutations
        for n, element in enumerate(all_routes.copy()):
            l = list(all_routes[n])
            l.insert(0, 0)
            l.append(0)
            all_routes[n] = tuple(l)
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
    random_route = random.sample(range(route_length), route_length)
    random_route.append(random_route[0])  # Add home travel
    return random_route

def create_route(route_length=6):
    route = list(range(route_length))
    return route

def get_total_distance(data, route) -> float:
    """
    Sum up the total distance for a route
    :param data: List of data
    :param route: Selected route
    :return: Total route distance
    """
    total_dist = 0
    # loop through route
    for step, travel in enumerate(route):
        try:
            dist = get_distance_cities(data, data[0][travel], data[0][route[step + 1]])
            total_dist += dist
        # Break off when reaching the end of the index
        except IndexError:
            break
    return round(total_dist, 2)
