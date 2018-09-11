"""
Author - Torstein Gombos
Created: 06.09.2018
Solving the travelling salesman problem
"""
import time
import csv
import argparse
import statistics
from oblig1 import routes as r
from oblig1 import simple_search_algorithms as search


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', '-m', type=str,
                        help='Pick a method for solving travelling salesman problem:\n'
                             'Current methods available\n:'
                             ' -m ex    -   Exhaustive search\n'
                             ' -m hc    -   Hill climber search')
    parser.add_argument('--route_length', '-r', type=int, default=10,
                        help='Choose length of route.')

    return parser.parse_args()


def read_csv(file) -> list:
    """
    Reads the content of a CSV file and store it to a data variable
    :param file: CSV file
    :return: List of data
    """
    with open(file, "r") as f:
        data = list(csv.reader(f, delimiter=';'))
    return data


def get_result(data, route_idx, travel_distances, algorithm):
    """
    Prints result from algorithm
    :param data: Data from CSV
    :param route_idx: The route. Should be index numbers
    :param travel_distances: The total distance of route
    :param algorithm: What algorithm was used
    :return: None
    """
    if algorithm == "exhaustive search":
        print("The shortest route found using {}:".format(algorithm))
        for city in route_idx:
            print(data[0][city], end=" ")
        print("\nThe total distance is {}km".format(travel_distances))

    # Calculating mean, standard deviation and best and worst performance
    elif algorithm == "hill climb":
        mean = sum(travel_distances) / len(travel_distances)
        std = statistics.stdev(travel_distances)
        shortest_dist, shortest_route = min(travel_distances), travel_distances.index(min(travel_distances))
        longest_dist, longest_route = max(travel_distances), travel_distances.index(max(travel_distances))


        print("The shortest route was {}km:".format(shortest_dist))
        for n, city in enumerate(route_idx[shortest_route]):
            print(data[0][city], "->", end=" ")
        print("\n\nThe longest route was {}km:".format(longest_dist))
        for city in route_idx[longest_route]:
            print(data[0][city], "->", end=" ")
        print("\n\nThe mean was: ", mean)
        print("The standard deviation was: ", std)




def main():
    # Read file and fetch data from csv file and parse arguments
    data = read_csv(file="european_cities.csv")
    args = parse_arguments()

    # Run exhaustive search
    if args.method == "ex":
        if args.route_length > 10:
            args.route_length = 10
        travel_distance, best_route = search.exhaustive_search(r.get_total_distance, data, args.route_length)
        get_result(data, best_route, travel_distance, algorithm="exhaustive search")

    # Run hill climber search
    elif args.method == "hc":
        print("Performing hill climber search for solving the travelling salesman problem:\n"
              "===========================================================================")
        travel_distances, best_routes = [], []
        for x in range(20):
            travel_distance, best_route = search.hill_climber(data, args.route_length, num_of_rand_resets=1)
            travel_distances.append(travel_distance), best_routes.append(best_route)

        get_result(data, best_routes, travel_distances, algorithm="hill climb")

    elif args.method == "ga":
        travel_distances, best_routes = search.genetic_algorithm(data, args.route_length)


# Time the execution
t0 = time.time()
main()
t1 = time.time()
print("\nCode execution: {}s".format(t1-t0))