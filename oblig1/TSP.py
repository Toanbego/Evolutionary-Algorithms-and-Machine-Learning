"""
Author - Torstein Gombos
Created: 06.09.2018
Solving the travelling salesman problem
"""
import time
import csv
import itertools
import random
from oblig1 import routes as r
# from Exercises import simple_search_algorithms as search
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

def exhaustive_search(data, search_space):
    """
    Function that searches every possible solution and returns global minimum
    :param f: Function
    :return: Returns y and x value
    """
    # Arbitrary start value
    max_value = r.get_total_distance(data, search_space[0])
    for step in search_space:
        new_value = r.get_total_distance(data, step)
        if new_value < max_value:
            max_value = new_value
            x_value = step
    return max_value, x_value

def get_result(data, route_idx, travel_distance, algorithm):
    """
    Prints result from algorithm
    :param data: Data from CSV
    :param route_idx: The route. Should be index numbers
    :param travel_distance: The total distance of route
    :param algorithm: What algorithm was used
    :return: None
    """
    route = []
    print("The shortest route using {}:".format(algorithm))
    for city in route_idx:
        print(data[0][city], end=" ")
        # route.append(data[0][city])
    print("\nThe total distance is {}km".format(travel_distance))

def hill_climber(data):
    """
    Hill climber algorithm that will check a neighboring solution
    If the neighbor solution is better, this becomes the new solution
    if not, keep the old one.
    :param data:
    :param search_space:
    :return:
    """
    # Set up a route
    route = r.create_route()
    precision = 0.01

    # Scramble route for randomness
    solution = random.sample(route, 24)

    while solution < precision:
        r.get_total_distance(data, solution)



def main():
    # Read file and fetch data from csv file
    file = "european_cities.csv"
    data = read_CSV(file)

    # Define routes
    # route = r.create_random_route(route_length=6)
    # routes = r.create_permutation_of_routes(route_length=6)

    # Use optimization algorithm
    # travel_distance, route_idx = exhaustive_search(data, routes)
    hill_climber(data)

    # Print result
    # get_result(data, route_idx, travel_distance, algorithm="exhaustive search")




# Time the function
t0 = time.time()
main()
t1 = time.time()
print("\nCode execution: {}s".format(t1-t0))