# Import
import csv
from itertools import permutations, combinations
import time
import numpy as np
import random
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def main():
    """
    The main loop, where the different algorithms can chosen
    """
    # Takes the time spent on exhaustive search
    # e0 = time.time()
    # exhaustive_search(csv, num_cities=9)
    # e1 = time.time()
    # print("Time spent on exhaustive search:", round((e1 - e0), 5), "sec \n")
    # h0 = time.time()
    # hill_climbing(csv, num_cities=24, iterations=20, plotting=True)
    # h1 = time.time()
    # print("Time spent on hill climbing:", round((h1 - h0), 5), "sec")
    g0 = time.time()
    genetic_algorithm(csv, num_cities=24, num_population=500, generations=1000, plotting=True)
    g1 = time.time()
    print("Time spent on genetic algorithm:", round((g1 - g0), 5), "sec")


with open("european_cities.csv", "r") as f:
    csv = list(csv.reader(f, delimiter=';'))


def exhaustive_search(csv_file, num_cities):
    """
    Exhaustive search algorithm, using permutations to find the guaranteed shortest possible route, max 10 cities
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
    """
    Hill climbing algorithm, swaps two cities at the time, and keeps the new route if distance is shorter.
    This algorithm will find a local/global optimum
    :param csv_file: csv file with the european cities
    :param num_cities: How many cities you want to run the algorithm for
    :param iterations: How many times you want the hill climber to run
    :param plotting: Plots the shortest distance of each run, and the standard deviation
    :return: None
    """
    best_of_best = []
    # Running hill climbing n times
    for iteration in range(iterations):
        start = random.sample(range(num_cities), num_cities) # Random start route
        combos = [start]
        # Makes a list of all possible combinations of two cities excluding start city
        combination = list(combinations(list(range(1, num_cities)), 2))
        # Start distance, route and combination
        best_dist, best_route, best_combo, _ = distance(combos, csv_file, num_cities)

        i = 0
        # Iterating trough all combinations, and resets if a swap of two cities gives shorter distance
        while i < len(combination):
            for idx, val in enumerate(combination):
                start[val[0]], start[val[1]] = start[val[1]], start[val[0]]  # Swapping two cities
                new_dist, new_route, new_combo, _ = distance(combos, csv_file, num_cities)  # New distance
                # If new distance is better than current best, then new distance is the current best
                if new_dist < best_dist:
                    best_dist, best_route, best_combo = new_dist, new_route, new_combo  # Sets new as best distance
                    i = 0  # Resets the iterator
                # If new distance is worse than current best, then swap the two cities back
                else:
                    start[val[0]], start[val[1]] = start[val[1]], start[val[0]]  # Swap back
                    i += 1  # Incrementing the iterator
        best_of_best.append(best_dist)  # Keeping track of all time best distance
    # If plotting argument is true, then plot
    if plotting is True:
        plot(best_of_best, iterations)
    # Printing stuff to terminal
    print("The shortest distance for", num_cities, "cities and", iterations, "iterations:", min(best_of_best), "km")
    print("The longest distance for", num_cities, "cities and", iterations, "iterations:", max(best_of_best), "km")
    print("The mean distance for", num_cities, "cities and", iterations, "iterations:", round(np.mean(best_of_best), 3), "km")
    print("The standard deviation for", num_cities, "cities and", iterations, "iterations:", round(np.std(best_of_best), 3))
    return


def genetic_algorithm(csv_file, num_cities, num_population, generations, plotting):
    """
    Genetic algorithm using fitness proportional selection with windowing as parent selection.
    Using order crossover for breeding offspring/new population, and swap mutation on the children.
    Generational model, the entire set of μ parents is replaced by μ offspring.
    :param csv_file: csv file with the european cities
    :param num_cities: How many cities you want to run the algorithm for
    :param num_population: How large the population is
    :param generations: How many generations to make
    :param plotting:
    :return: None
    """
    # Creates a random initialization population
    population = create_population(num_cities, num_population)
    generation_best = []
    # Loop for n generations
    for generation in range(generations):
        best, route, combo, fitness = distance(population, csv_file, num_cities)
        parents = parent_selection(num_population, fitness, population)  # Parent selection
        offspring = breed_population(parents)  # Breeding based on recombination (crossover)
        # Doing swap mutation on children
        for i in range(len(offspring)):
            if random.random() < 1/num_cities:  # Mutating based on a small probability
                swap_mutation(offspring[i])  # swapping two random elements of all individuals
        population = offspring  # The offspring is now the new population
        generation_best.append(best)  # Saving best distance for each generation
    # Plotting
    if plotting is True:
        plt.plot(generation_best)
        plt.show()
    print("Shortest distance is", min(generation_best), "km in generation", generation_best.index(min(generation_best)))
    return


def parent_selection(num_population, fitness, population):
    """
    Parent selection based on fitness proportional selection with windowing to avoid premature
    convergence and make selection pressure.
    :param num_population: How large the population is
    :param fitness: The fitness of the individuals in the population
    :param population: The population, i.e the individuals
    :return: Mating pool containing the parents
    """
    # Using windowing to avoid premature convergence and make selection pressure
    window_fitness = windowing(fitness)
    fitness_proportional = []
    # Fitness proportional selection after windowing
    for i in range(num_population):
        fitness_proportional.append(window_fitness[i]/sum(window_fitness))  # fitness(i)/sum(fitness)
    # Finding the cumulative probability of the fps
    fitness_proportional = np.cumsum(fitness_proportional)

    # This part is based on Fig. 5.2. in Introduction to evolutionary computing p.85
    # Stochastic universal sampling algorithm making λ selections
    # Equivalent to making one spin of a wheel with λ equally spaced arms
    mating_pool = []
    pool_size = num_population
    current_member = i = 0
    r = random.uniform(0.0, (1 / pool_size))
    while current_member < pool_size:
        while r < fitness_proportional[i]:
            mating_pool.append(population[i])
            r += 1 / pool_size
            current_member += 1
        i += 1
    return mating_pool


def order_crossover(parent1, parent2):
    """
    Using order crossover as recombination.
    Copy randomly selected segment from parent into offspring.
    Copy rest of alleles in order they appear in second parent.
    Reverse the parents to make the seconds children.
    :param parent1: Parent 1 for children
    :param parent2: Parent 2 to the children
    :return: Child (one individual)
    """
    # Making two lists with None elements
    child = [None] * len(parent1)
    # Generating two random integers between the value of the length of the parents

    element1, element2 = random.randint(0, len(parent1)), random.randint(0, len(parent2))
    # Setting min value as start and max value as end
    start_gene, end_gene = min(element1, element2), max(element1, element2)
    # Step 1 order crossover: copy randomly selected segment from parent into offspring
    for i in range(start_gene, end_gene):
        child[i] = parent1[i]

    # step 2: copy rest of alleles in order they appear in second parent, treating string as toroidal
    i = end_gene
    for idx in range(len(child)):
        # Checks if element in parent2 is not in child1
        if parent2[(idx + end_gene) % len(parent2)] not in child:
            # If not, then place it in child1
            child[i % len(child)] = parent2[(idx + end_gene) % len(parent2)]
            i += 1
    return child


def breed_population(mating_pool):
    """
    Breeding population using order crossover on individuals in the mating pool
    :param mating_pool: Containing all the parents
    :return: Offspring, i.e the new population
    """
    offspring = []
    p1, p2 = 0, 1
    # Doing order crossover for each parent in the mating pool
    while p2 < len(mating_pool):
        offspring.append(order_crossover(mating_pool[p1], mating_pool[p2]))  # child1
        offspring.append(order_crossover(mating_pool[p2], mating_pool[p1]))  # child2
        p1 += 2
        p2 += 2
    return offspring


def create_population(num_cities, num_population):
    """
    Makes a random start population
    :param num_cities: How many cities you want to run the algorithm for
    :param num_population: How large the population is
    :return: population
    """
    population = []
    # Making individuals for the population
    for i in range(num_population):
        population.append(random.sample(range(num_cities), num_cities))  # Random individual
    return population


def windowing(fitness):
    """
    Windowing based on the fitness of the population
    :param fitness: distances of all routes in the population
    :return: windowed fitness
    """
    window = []
    # The worst distance in the population
    worst_fitness = max(fitness)
    # Windowing: (fitness(i) - worst fitness)
    for i in fitness:
        window.append(abs(i - worst_fitness))
    return window


def swap_mutation(individual):
    """
    Swaps two random cities in the individual
    :param individual: one route
    :return: None
    """
    idx = range(len(individual))
    idx1, idx2 = random.sample(idx, 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return


def plot(data, iterations):
    """
    Plot for hill climbing, plotting best distance of each iteration
    :param data: Best distances
    :param iterations: How many times the hill climber ran
    :return: None
    """
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
    """
    Finds the distance of either one or more routes in an array.
    Finds the best distance of different routes and the associated route
    :param combos: 1 or more routes
    :param csv_file: csv file with the european cities
    :param num_cities: How many cities you want to run the algorithm for
    :return: Shortest distance, which route it is (names), which route in numbers, all distances if several routes
    """
    # Appends all distances for each travel in each permutation in a list
    dist = []
    for element in combos:
        for idx, val in enumerate(element):
            dist.append(float(csv_file[1+element[idx]][element[idx-1]]))
    # Sums travel distances for each route in a list
    all_distances = list(np.add.reduceat(dist, np.arange(0, len(dist), num_cities)))
    # Finds the shortest distance in the list, and the corresponding index
    shortest_distance, index = round(min(all_distances), 3), all_distances.index(min(all_distances))
    # Finding the cities corresponding to the best combo, i.e the travel route
    best_combo, best_route = combos[index], []
    # print(best_combo)
    # print(distances)
    for city in best_combo:
        best_route.append(str(csv_file[0][city]))

    # Just to show that it ends in the same city
    best_route.append(str(csv_file[0][best_combo[0]]))

    return shortest_distance, best_route, best_combo, all_distances


if __name__ == "__main__":
    main()


