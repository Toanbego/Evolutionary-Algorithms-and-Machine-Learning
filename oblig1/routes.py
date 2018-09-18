"""
Author - Torstein Gombos
Created - 06.09.2018

Package with functions regarding routes and distance for routes
"""
import random
import itertools
import numpy as np


def get_distance_cities(data: list, city1: str, city2: str) -> float:
    """
    Calculates the distance between two cities from a CSV file
    Order of city does not matter.
    :param data: The data from the CSV file
    :param city1: string with city name
    :param city2: string with city name
    :return:
    """
    return float(data[data[0].index(city1)+1][data[0].index(city2)])



def create_permutation_of_routes(route_length=6) -> list:
    """
    Create permutation of a set of cities
    User provides how many cities to be included
    :param route_length: How many cities to be included
    :param random_route: Checks for a random set of cities of the 24 cities
    :return: List of all permutations of the set of cities
    """
    route_starting_point = list(range(route_length))
    random.shuffle(route_starting_point)
    all_routes = list(itertools.permutations(route_starting_point[1:]))

    # Appends home destination to all permutations
    for n, element in enumerate(all_routes.copy()):
        list_of_routes = list(all_routes[n])
        list_of_routes.insert(0, route_starting_point[0])
        list_of_routes.append(route_starting_point[0])
        all_routes[n] = tuple(list_of_routes)
    return all_routes


def create_random_route(route_length=10, seed=random.random()):
    """
    Returns a random sequence that can be used to access different indexes
    in the data variable from the CSV file.
    :param route_length: Length of sequence
    :return:
    """
    # Generate a random route sequence.
    random.seed(seed)
    random_route = random.sample(range(route_length), route_length)  # Generate random route
    shuffle_inside = random_route[1:]
    random.seed()
    random.shuffle(shuffle_inside)
    random_route[1:] = shuffle_inside
    random_route.append(random_route[0])  # Add home travel
    return random_route


def create_random_route_from_first_n_cities(route_length=10):
    """
    Creates a random route containing the n first cities in the
    CSV file. Used to get right results from hill climber
    :param route_length:
    :return:
    """
    route = list(range(route_length))
    random.shuffle(route)
    route.append(route[0])
    return route


def get_total_distance(data: list, route: list) -> float:
    """
    Sum up the total distance for a route
    :param data: List of data
    :param route: Selected route
    :return: Total route distance
    """
    dist_list = [get_distance_cities(data, data[0][route[step]], data[0][route[step + 1]])
                 for step in range(len(route)-1)]
    dist = sum(dist_list)
    return round(dist, 2)
