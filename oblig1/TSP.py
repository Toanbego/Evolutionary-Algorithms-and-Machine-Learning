"""
Author - Torstein Gombos
Created: 06.09.2018
Solving the travelling salesman problem

This is the main script, which does very little by itself. It will
set up and run the various methods based on users arguments. See readme file for how
to do that. The main function is hard coded to give out answers to the questions in the assignment.
Anything related to optimization algorithms is imported from the
simple_search_algorithms.py script
anything related to creating routes and calculating distances is
imported from routes.py

"""
import time
import csv
import argparse
import statistics
import matplotlib.pyplot as plt
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
    """
    Plots result from genetic algorithm and hybrid
    :param results:
    :param population_sizes:
    :param size:
    :return:
    """
    mean_best_fit = []
    print(len(results[:]))
    print(len(results[:][0]))
    print(results[:][0])
    for n in range(len(results[0])):
        mean_best_fit.append(statistics.mean(results[:][n]))

    if size == population_sizes[0]:
        label1, = plt.plot(range(0, len(result)), mean_best_fit[:], label=str(population_sizes[0]))
    elif size == population_sizes[1]:
        label2, = plt.plot(range(0, len(result)), mean_best_fit[:], label=str(population_sizes[1]))
    elif size == population_sizes[2]:
        label3, = plt.plot(range(0, len(result)), mean_best_fit[:], label=str(population_sizes[2]))

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

        # Run hill climber and time it
        t0 = time.time()
        travel_distances, best_route = search.hill_climber(data, args.route_length, first_ten=True)
        t1 = time.time()

        # Print out results
        print("The shortest route:")
        for city in best_route:
            print(data[0][city], "->", end=" ")
        print("\nThe total distance is {}km".format(travel_distances))
        print("\nCode execution: {}s".format(t1 - t0))
        print("\n   -------------------------------------")
        print("\nPerforming 20 hill climbs on random sequence of {} cities: ".format(args.route_length))
        travel_distances, best_routes = [], []

        # Print results from 20 runs
        for x in range(20):
            travel_distance, best_route = search.hill_climber(data, args.route_length)
            travel_distances.append(travel_distance), best_routes.append(best_route)
        get_result(data, best_routes, travel_distances, algorithm="hill climb")

    # Run genetic algorithm
    elif args.method == "ga":
        # TODO gjennmsnitt av beste fitness for hver generasjon
        # Set up population sizes
        population_sizes = [500, 700, 1200]
        for size in population_sizes:
            fitnesses = []
            last_fitnesses = []
            best_route = []
            worst_route = []
            tid0 = time.time()

            # Perform GA 20 times
            print("\nPerforming with ", size)
            for i in range(5):
                print("Run {} with {}".format(i, size))

                # Start genetic algorithm
                best_fitness, last_fitness, population, evals =\
                    search.genetic_algorithm(data, args.route_length, size)

                # Calculate and store results
                tid1 = time.time()
                fitnesses.append(best_fitness)  # All best fitness for each generations
                last_fitnesses.append(last_fitness)  # Best fitness from last generation
                best_route.append(population[evals.index(max(evals))])
                worst_route.append(population[evals.index(min(evals))])

            # Print out results
            print("Ga execution time: ", (tid1 - tid0))
            print("The shortest route was {}km:".format(min(last_fitnesses)))
            for city in best_route[last_fitnesses.index(min(last_fitnesses))]:
                print(data[0][city], "->", end=" ")
            print("\n\nThe longest route was {}km:".format(max(last_fitnesses)))
            for city in worst_route[last_fitnesses.index(max(last_fitnesses))]:
                print(data[0][city], "->", end=" ")
            if i > 2:
                average_20_runs = statistics.mean(last_fitnesses),  # Average fitness for 20 runs
                std_20_runs = statistics.stdev(last_fitnesses)  # Standard deviation for fitness for 20 runs
                print("\n\nThe mean was: ", average_20_runs)
                print("The standard deviation was: ", std_20_runs)

            # Plot results
            # plot(fitnesses[last_fitnesses.index(min(last_fitnesses))], population_sizes, size)
            plot(fitnesses, population_sizes, size)
        plt.savefig('{} - {}.png'.format(args.route_length, time.time()))

    # Run hybrid algorithm
    elif args.method == "hybrid":

        # Set up population sizes
        population_sizes = [500, 700, 1200]
        for size in population_sizes:
            fitnesses = []
            last_fitnesses = []
            best_route = []
            worst_route = []

            # Run Hybrid 20 times
            print("\nPerforming with ", size)
            for i in range(3):
                print("Run {} with {}".format(i, size))

                # Perform hybrid genetic algorithm
                best_fitness, last_fitness, population, evals =\
                    search.genetic_algorithm(data, args.route_length, size, hybrid=True)

                # Calculate and store results
                fitnesses.append(best_fitness)  # All best fitness for each generations
                last_fitnesses.append(last_fitness)  # Best fitness from last generation
                best_route.append(population[evals.index(max(evals))])
                worst_route.append(population[evals.index(min(evals))])

            average_20_runs = statistics.mean(last_fitnesses),  # Average fitness for 20 runs
            std_20_runs = statistics.stdev(last_fitnesses)  # Standard deviation for fitness for 20 runs

            # Print results
            print("The shortest route was {}km:".format(min(last_fitnesses)))
            for city in best_route[last_fitnesses.index(min(last_fitnesses))]:
                print(data[0][city], "->", end=" ")
            print("\n\nThe longest route was {}km:".format(max(last_fitnesses)))
            for city in worst_route[last_fitnesses.index(max(last_fitnesses))]:
                print(data[0][city], "->", end=" ")
            print("\n\nThe mean was: ", average_20_runs)
            print("The standard deviation was: ", std_20_runs)

            # Plot Results
            plot(fitnesses[last_fitnesses.index(min(last_fitnesses))], population_sizes, size)
            # plot(fitnesses[last_fitnesses.index(min(last_fitnesses))], population_sizes, size)

        plt.show()


# Time the execution
t0 = time.time()
main()
t1 = time.time()
print("\nCode execution: {}s".format(t1-t0))