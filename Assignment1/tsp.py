# Import
import csv
from itertools import permutations
import time
import numpy as np
import random


with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))


def exhaustive_search(csv_file, num_cities):
    """
    Exhaustive search algorithm
    :param csv_file: csv file with the european cities
    :param num_cities: How many cities you want to run the algorithm for
    :return: None
    """
    # Makes a list of all possible routes with n cities
    combos = list(permutations(list(range(num_cities))))
    # Appends all distances for each travel in each permutation in a list
    dist = []
    print(len(combos))
    for element in combos:
        for idx, val in enumerate(element):
            dist.append(float(csv_file[1+element[idx]][element[idx-1]]))
    # Sums travel distances for each route in a list
    distances = list(np.add.reduceat(dist, np.arange(0, len(dist), num_cities)))
    # Finds the shortest distance in the list, and the corresponding index
    shortest_distance, index = min(distances), distances.index(min(distances))
    # Finding the cities corresponding to the best combo, i.e the travel route
    best_combo, best_route = combos[index], []
    # print(best_combo)
    # print(distances)
    for city in best_combo:
        best_route.append(str(csv_file[0][city]))

    # Just to show that it ends in the same city
    best_route.append(str(csv_file[0][best_combo[0]]))
    # Printing to the terminal
    print("Shortest travel route for", num_cities, "cities:", best_route)
    print("Shortest travel distance for", num_cities, "cities:", round(shortest_distance, 3), "km")
    return


def hill_climbing(csv_file, num_cities):
    start = tuple(random.sample(range(num_cities), num_cities))
    #     print(csv_file[1 + start[i]][start[i - 1]])
    best = []
    for idx, val in enumerate(start):
        print(csv_file[1 + start[idx]][start[idx - 1]])
        best.append(str(val))
    # print(best)


    return


def genetic_algorithm():
    pass


def hybrid_algorithm():
    pass


def main():
    # Takes the time spent on exhaustive search
    # t0 = time.time()
    # exhaustive_search(data, num_cities=9)
    # t1 = time.time()
    # print("Time spent on exhaustive search:", round((t1 - t0), 5), "sec")
    hill_climbing(data, num_cities=6)


if __name__ == "__main__":
    main()


