# Import
import csv
from itertools import permutations, combinations
import time
import numpy as np
import random
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def main():
    # Takes the time spent on exhaustive search
    e0 = time.time()
    exhaustive_search(csv, num_cities=9)
    e1 = time.time()
    print("Time spent on exhaustive search:", round((e1 - e0), 5), "sec \n")
    # h0 = time.time()
    # hill_climbing(csv, num_cities=24, iterations=20, plotting=True)
    # h1 = time.time()
    # print("Time spent on hill climbing:", round((h1 - h0), 5), "sec")
    # genetic_algorithm(csv, num_cities=10, num_population=10)


with open("european_cities.csv", "r") as f:
    csv = list(csv.reader(f, delimiter=';'))


def exhaustive_search(csv_file, num_cities):
    """
    Exhaustive search algorithm
    :return:
    :param csv_file: csv file with the european cities
    :param num_cities: How many cities you want to run the algorithm for
    :return: None
    """
    # Makes a list of all possible routes with n cities
    combos = list(permutations(range(num_cities)))

    # Runs the distance function
    shortest_distance, best_route, best_combo, _ = distance(combos, csv_file, num_cities)
    print("Shortest travel route for", num_cities, "cities:", best_route)
    print("Shortest travel distance for", num_cities, "cities:", round(shortest_distance, 3), "km")
    return


def hill_climbing(csv_file, num_cities, iterations, plotting):

    best_of_best = []
    for iteration in range(iterations):
        start = random.sample(range(num_cities), num_cities)
        combos = [start]
        combination = list(combinations(list(range(1, num_cities)), 2))
        best_dist, best_route, best_combo, _ = distance(combos, csv_file, num_cities)

        i = 0
        while i < len(combination):
            for idx, val in enumerate(combination):
                start[val[0]], start[val[1]] = start[val[1]], start[val[0]]

                new_dist, new_route, new_combo, _ = distance(combos, csv_file, num_cities)
                if new_dist < best_dist:
                    best_dist, best_route, best_combo = new_dist, new_route, new_combo
                    i = 0
                else:
                    start[val[0]], start[val[1]] = start[val[1]], start[val[0]]
                    i += 1
        best_of_best.append(best_dist)
    if plotting is True:
        plot(best_of_best, iterations)

    print("The shortest distance for", num_cities, "cities and", iterations, "iterations:", min(best_of_best), "km")
    print("The longest distance for", num_cities, "cities and", iterations, "iterations:", max(best_of_best), "km")
    print("The mean distance for", num_cities, "cities and", iterations, "iterations:", round(np.mean(best_of_best), 3), "km")
    print("The standard deviation for", num_cities, "cities and", iterations, "iterations:", round(np.std(best_of_best), 3))
    return


def genetic_algorithm(csv_file, num_cities, num_population):
    population = create_population(num_cities, num_population)
    _, _, _, distances = distance(population, csv_file, num_cities)
    print(distances)

    return


def hybrid_algorithm():
    pass


def create_population(num_cities, num_population):
    population = []
    for i in range(num_population):
        population.append(random.sample(range(num_cities), num_cities))
    return population


def plot(data, iterations):
    mean = []
    for x in range(len(data)):
        mean.append(np.mean(data))

    plt.plot(range(0, len(data)), data, 'ro', color='black')
    plt.plot(range(0, len(data)), mean, color='red', label="mean")
    plt.plot(range(0, len(data)), mean + np.std(data), color="blue", label='std')
    plt.plot(range(0, len(data)), mean - np.std(data), color="blue")
    plt.legend()
    plt.xlim(-1, iterations)
    plt.xlabel('Number of iterations')
    plt.ylabel('Best distance for each iteration (Km)')
    plt.show()
    return


def distance(combos, csv_file, num_cities):
    # Appends all distances for each travel in each permutation in a list
    dist = []
    for element in combos:
        for idx, val in enumerate(element):
            dist.append(float(csv_file[1+element[idx]][element[idx-1]]))
    # Sums travel distances for each route in a list
    distances = list(np.add.reduceat(dist, np.arange(0, len(dist), num_cities)))
    # Finds the shortest distance in the list, and the corresponding index
    shortest_distance, index = round(min(distances), 3), distances.index(min(distances))
    # Finding the cities corresponding to the best combo, i.e the travel route
    best_combo, best_route = combos[index], []
    # print(best_combo)
    # print(distances)
    for city in best_combo:
        best_route.append(str(csv_file[0][city]))

    # Just to show that it ends in the same city
    best_route.append(str(csv_file[0][best_combo[0]]))
    return shortest_distance, best_route, best_combo, distances


if __name__ == "__main__":
    main()


