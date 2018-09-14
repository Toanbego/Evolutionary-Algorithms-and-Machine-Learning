"""
Author - Torstein Gombos
Created: 06.09.2018
Solving the travelling salesman problem
"""
import time
import csv
import argparse
import statistics
import matplotlib.pyplot as plt
import numpy as np
from oblig1 import simple_search_algorithms as search
from oblig1 import routes as r



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
    parser.add_argument("--learning_model", "-l", type=str, default="lamarck",
                        help='Choose either lamarckian or baldwinian learning method'
                             'Only usable for hybrid method')

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


def plot(results, population_sizes, size):

    if size == population_sizes[0]:
        label1, = plt.plot(range(0, len(results)), results[:], label=str(population_sizes[0]))
    elif size == population_sizes[1]:
        label2, = plt.plot(range(0, len(results)), results[:], label=str(population_sizes[1]))
    elif size == population_sizes[2]:
        label3, = plt.plot(range(0, len(results)), results[:], label=str(population_sizes[2]))

        plt.title('Genetic Algorithm')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.legend([str(population_sizes[0]),
                    str(population_sizes[1]),
                    str(population_sizes[2])],
                   loc='upper right')





def main():
    """
    This main function simply checks arguments and runs the various
    solutions and print the results according to the questions in the assignment
    :return:
    """
    # Read file and fetch data from csv file and parse arguments
    data = read_csv(file="european_cities.csv")
    args = parse_arguments()

    # Run exhaustive search
    if args.method == "ex":
        if args.route_length > 11:  # User may not use longer routes than 11 cities
            args.route_length = 11
        travel_distance, best_route = search.exhaustive_search(data, args.route_length)
        get_result(data, best_route, travel_distance, algorithm="exhaustive search")

    # Run hill climber search
    elif args.method == "hc":
        print("-- HILL CLIMBING --\n")
        print("Performing on first 10 cities:")
        t0 = time.time()
        travel_distances, best_route = search.hill_climber(data, args.route_length, first_ten=True)
        t1 = time.time()
        print("The shortest route:")
        for city in best_route:
            print(data[0][city], "->", end=" ")
        print("\nThe total distance is {}km".format(travel_distances))
        print("\nCode execution: {}s".format(t1 - t0))
        print("\n   -------------------------------------")
        print("\nPerforming 20 hill climbs on random sequence of {} cities: ".format(args.route_length))
        travel_distances, best_routes = [], []
        for x in range(50):
            travel_distance, best_route = search.hill_climber(data, args.route_length)
            travel_distances.append(travel_distance), best_routes.append(best_route)
        get_result(data, best_routes, travel_distances, algorithm="hill climb")

    # Run genetic algorithm

    elif args.method == "ga":
        population_sizes = [500, 700, 1200]
        for size in population_sizes:
            fitnesses = []
            last_fitnesses = []
            best_route = []
            worst_route = []
            tid0 = time.time()
            # Perform GA
            print("\nPerforming with ", size)
            for i in range(1):
                print("Run {} with {}".format(i, size))
                best_fitness, last_fitness, population, evals =\
                    search.genetic_algorithm(data, args.route_length, size, args.learning_model)
                tid1 = time.time()
                fitnesses.append(best_fitness)  # All best fitness for each generations
                last_fitnesses.append(last_fitness)  # Best fitness from last generation
                best_route.append(population[evals.index(max(evals))])
                worst_route.append(population[evals.index(min(evals))])

            # average_20_runs = statistics.mean(last_fitnesses),  # Average fitness for 20 runs
            # std_20_runs = statistics.stdev(last_fitnesses)  # Standard deviation for fitness for 20 runs
            print("Ga execution time: ", (tid1 - tid0))
            print("The shortest route was {}km:".format(min(last_fitnesses)))
            for city in best_route[last_fitnesses.index(min(last_fitnesses))]:
                print(data[0][city], "->", end=" ")
            print("\n\nThe longest route was {}km:".format(max(last_fitnesses)))
            for city in worst_route[last_fitnesses.index(max(last_fitnesses))]:
                print(data[0][city], "->", end=" ")
            # print("\n\nThe mean was: ", average_20_runs)
            # print("The standard deviation was: ", std_20_runs)

            # Plot
            plot(fitnesses[last_fitnesses.index(min(last_fitnesses))], population_sizes, size)
        plt.show()

    # Run hybrid algorithm

    elif args.method == "hybrid":
        population_sizes = [100, 700, 1200]
        for size in population_sizes:
            fitnesses = []
            last_fitnesses = []
            best_route = []
            worst_route = []

            # Perform GA
            print("\nPerforming with ", size)
            for i in range(3):
                print("Run {} with {}".format(i, size))
                best_fitness, last_fitness, population, evals =\
                    search.genetic_algorithm(data, args.route_length, size, hybrid=True)
                fitnesses.append(best_fitness)  # All best fitness for each generations
                last_fitnesses.append(last_fitness)  # Best fitness from last generation
                best_route.append(population[evals.index(max(evals))])
                worst_route.append(population[evals.index(min(evals))])

            average_20_runs = statistics.mean(last_fitnesses),  # Average fitness for 20 runs
            std_20_runs = statistics.stdev(last_fitnesses)  # Standard deviation for fitness for 20 runs

            print("The shortest route was {}km:".format(min(last_fitnesses)))
            for city in best_route[last_fitnesses.index(min(last_fitnesses))]:
                print(data[0][city], "->", end=" ")
            print("\n\nThe longest route was {}km:".format(max(last_fitnesses)))
            for city in worst_route[last_fitnesses.index(max(last_fitnesses))]:
                print(data[0][city], "->", end=" ")
            print("\n\nThe mean was: ", average_20_runs)
            print("The standard deviation was: ", std_20_runs)

            # Plot
            plot(fitnesses[last_fitnesses.index(min(last_fitnesses))], population_sizes, size)
        plt.show()



# Time the execution
t0 = time.time()
main()
t1 = time.time()
print("\nCode execution: {}s".format(t1-t0))