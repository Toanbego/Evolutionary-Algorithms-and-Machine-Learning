# Import
import csv
from itertools import combinations, permutations
import time
import numpy as np


with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))


def exhaustive_search(csv_file, num_cities):
    comb = list(permutations(list(range(num_cities))))
    dist = []
    for element in comb:
        for i in element:
            dist.append(float(csv_file[1+element[i]][element[i-1]]))

    distances = list(np.add.reduceat(dist, np.arange(0, len(dist), num_cities)))
    shortest_distance, index = min(distances), distances.index(min(distances))
    best_combo = comb[index]
    best_route = []
    for city in best_combo:
        best_route.append(str(csv_file[0][city]))
    best_route.append(str(csv_file[0][best_combo[0]]))
    print("Shortest travel route for", num_cities, "cities:", best_route)
    print("Shortest travel distance for", num_cities, "cities:", round(shortest_distance, 3), "km")
    return


def hill_climbing():
    pass


def genetic_algorithm():
    pass


def hybrid_algorithm():
    pass


def main():
    t0 = time.time()
    exhaustive_search(data, num_cities=6)
    t1 = time.time()
    print("Time spent on exhaustive search:", round((t1 - t0), 5), "sec")


if __name__ == "__main__":
    main()


