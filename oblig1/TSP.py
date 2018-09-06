"""
Author - Torstein Gombos
Created: 06.09.2018
Solving the travelling salesman problem
"""
import time
import csv
import itertools
import random
from Exercises import simple_search_algorithms as search
exhaustive_search = search.exhaustive_search()
# TODO Implement exhaustive search from previous assignment


def read_CSV(file) -> list:
    """
    Reads the content of a CSV file and store it to a data variable
    :param file: CSV file
    :return: List of data
    """
    with open(file, "r") as f:
        data = list(csv.reader(f, delimiter=';'))
    return data

def get_distance_cities(data, city1="Copenhagen", city2="Dublin") -> float:
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

def create_permutation_of_routes(data, route_length=6):
    """
    Create permutation of routes given length
    :param data:
    :param route_length:
    :return:
    """
    route_sequence = list(range(route_length))
    all_routes = list(itertools.permutations(route_sequence))
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
    random_route = list(range(route_length))
    random_route = random.sample(range(24), 6)
    return random_route

def get_total_distance(data, route):
    """
    Recieve total distance from the selected route
    :param data: List of data
    :param route: Selected route
    :return: Total route distance
    """
    total_dist = 0
    for step, travel in enumerate(route):
        try:
            dist = get_distance_cities(data, data[0][travel], data[0][route[step + 1]])
            total_dist += dist
        except IndexError:
            break
        print("distance from {} to {} is: {}".format(data[0][travel],
                                                     data[0][route[step + 1]], dist))
    print("Total route distance: ", total_dist)
    return round(total_dist, 2)

def main():
    # Read file and fetch data from csv file
    file = "european_cities.csv"
    data = read_CSV(file)

    route = create_random_route(route_length=6)
    routes = create_permutation_of_routes(data, route_length=6)

    exhaustive_search(get_total_distance(), routes)
    total_distance = get_total_distance(data, route)


t0 = time.time()
main()
t1 = time.time()
print("Code execution: ", t1-t0)