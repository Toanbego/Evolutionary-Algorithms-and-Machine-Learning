"""
Author - Torstein Gombos
Created: 06.09.2018
Solving the travelling salesman problem
"""
import time
import csv

from oblig1 import routes as r
from oblig1 import simple_search_algorithms as search

# TODO Make mutations and crossovers a class


def read_csv(file) -> list:
    """
    Reads the content of a CSV file and store it to a data variable
    :param file: CSV file
    :return: List of data
    """
    with open(file, "r") as f:
        data = list(csv.reader(f, delimiter=';'))
    return data


def get_result(data, route_idx, travel_distance, algorithm):
    """
    Prints result from algorithm
    :param data: Data from CSV
    :param route_idx: The route. Should be index numbers
    :param travel_distance: The total distance of route
    :param algorithm: What algorithm was used
    :return: None
    """
    print("The shortest route found using {}:".format(algorithm))
    for city in route_idx:
        print(data[0][city], end=" ")
    print("\nThe total distance is {}km".format(travel_distance))


def main():
    # Read file and fetch data from csv file
    file = "european_cities.csv"
    data = read_csv(file)

    # Use optimization algorithm
    # travel_distance, best_route = search.exhaustive_search(r.get_total_distance, data, route_length=10)
    travel_distance, best_route = search.hill_climber(data, route_length=10, num_of_rand_resets=100000)

    # Print result
    get_result(data, best_route, travel_distance, algorithm="hill climb")
    # get_result(data, best_route, travel_distance, algorithm="exhaustive search")


# Time the execution
t0 = time.time()
main()
t1 = time.time()
print("\nCode execution: {}s".format(t1-t0))