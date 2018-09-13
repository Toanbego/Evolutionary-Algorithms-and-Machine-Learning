"""
Author - Torstein Gombos
Created - 06.09.2018

Module with some simple search algorithms that can be used for optimization.
"""
import random
import numpy as np
from oblig1 import routes as r
import statistics



class Population:
    """
    A class representing a populations of solutions and their
    mutations and crossovers over generations
    """
    def __init__(self, data,  population, generation=0, eliteism=False):
        """
        Initiate population and evaluate it
        :param data:
        :param population:
        :param generation:
        """

        # Parameters and data
        self.generation = generation
        self.pmx_prob = 0.8  # 80% chance for pmx crossover in offspring
        self.mutation_prob = 1/len(population[0])  # 1/number_of_cities
        self.data = data  # TSP matrix
        self.eliteism = eliteism

        # Set up population and evaluate
        self.population = population  # Initialize population
        self.evaluation = self.evaluate_population(self.population)  # Evaluate current population
        if self.eliteism:  # If true, choose a set of elites from the 10% best of population
            self.elites = sorted(range(len(self.evaluation)),
                                                 key=lambda i: self.evaluation[i])[:int(len(self.population) * 0.1)]

        # Create new population. Initiated in evolve method
        self.parents = None
        self.offsprings = None  # Create offsprings
        self.evaluated_offspring = None

    def evaluate_population(self, population):
        """
        Evaluate the fitness of a given population
        :param population: A list with permutations of routes.
        :return: Fitness
        """
        # Subtract fitness from 100000 is a trick to make shortest distance best fitness
        fitness = [(100000 - r.get_total_distance(self.data, individual)) for individual in population]
        return fitness

    def select_parents(self):
        """
        Selects parents based on fitness proportionate selection.
        Performs a windowing on fitness beforehand ot increase
        the chance for better solutions to be picked.
        :return:
        """
        # Perform windowing on fitness
        window_fitness = [(self.evaluation[i] - min(self.evaluation)) for i, item in enumerate(self.evaluation)]

        # Create selection wheel
        selection_wheel = [(fitness/sum(window_fitness)) for fitness in window_fitness]
        selection = np.random.choice(range(len(self.population)), len(selection_wheel),
                                     p=selection_wheel)  # Random selection
        parents = [self.population[parent] for parent in selection]
        return parents

    def evolve(self):
        """
        Method for evolving generation by selecting parents and creating
        offsprings. Usually called outside class
        :return:
        """
        self.parents = self.select_parents()  # Select new parents
        self.offsprings = self.pmx(self.pmx_prob)  # Create offsprings through pmx crossover
        self.evaluated_offspring = self.evaluate_population(self.offsprings)  # Evaluate the offsprings
        self.mutate_population(self.offsprings, self.mutation_prob)  # Mutates the new population
        self.replace_population()
        self.generation += 1

        return self.population, self.generation

    def pmx(self, prob=0.8):
        """
        Performs pmx on as many parents as the prob variable of parents
        :return:
        """
        offsprings = []
        # Loop through mating pool
        for i in range(0, len(self.parents), 2):

            # Probability that pmx happens
            if random.random() < prob:
                offsprings.append(self.parents[i])
                offsprings.append(self.parents[i + 1])
                continue

            # For loop to create two offsprings from same parents
            for n in range(2):
                # Initiate offspring and two sub-segments
                if n == 0:
                    parent1, parent2 = self.parents[i], self.parents[i + 1]
                    start_stop = random.sample(range(1, len(parent1) - 1), 2)
                    start, stop = min(start_stop), max(start_stop)
                    sub_segment1, sub_segment2 = parent1[start:stop], parent2[start:stop]
                elif n == 1:
                    parent2, parent1 = self.parents[i+1], self.parents[i]
                    sub_segment1, sub_segment2 = parent2[start:stop], parent1[start:stop]
                offspring = [None] * (len(parent1))
                offspring[start:stop] = sub_segment1

                # Check if first element in sub_segment2 is in offspring1
                for element in sub_segment2:
                    # Copy elements from sub_segment2 to offspring1
                    if element not in offspring:
                        idx = parent2.index(element) # Find position for element in p2
                        while offspring[idx] is not None:
                            value_for_idx_in_p1 = parent1[idx]  # Find the value for the same position in p1
                            idx = parent2.index(value_for_idx_in_p1)  # Find the position for p1 in p2
                            if offspring[idx] is None:  # Check if not occupied
                                offspring[idx] = element  # Copy element to this position
                                break

                # Copy remaining numbers from parent2 onto offspring
                for z, element in enumerate(parent2):
                    if offspring[z] is None:
                        offspring[z] = element
                offsprings.append(offspring)

        return offsprings

    def mutate_population(self, population, mutation_prob=1/24):
        """
        A probability that an individual will mutate with a swap permutation
        :return:
        """
        self.offsprings = population
        mutated_population = []
        for i, individual in enumerate(population):
            try:
                if random.random() < mutation_prob:
                    seq_idx = list(range(len(individual)))
                    a1, a2 = random.sample(seq_idx[1:-1], 2)
                    individual[a1], individual[a2] = individual[a2], individual[a1]
                    #Updates offspring
                    self.offsprings[i] = individual
                else:
                    continue
            except TypeError:
                break

    def replace_population(self, eliteism=False):
        """
        Returns a new population
        Selects a group of the best from current population (eliteism)
        Selects the best new offsprings
        Replaces the worst of the current population
        :return:
        """
        if not eliteism:
            self.population = self.offsprings
        else:
            best_population = sorted(range(len(self.evaluation)),
                                             key=lambda i: self.evaluation[i])[:int(len(self.population) * 0.1)]
            best_offsprings = sorted(range(len(self.evaluated_offspring)),
                                    key=lambda i: self.evaluated_offspring[i])[:int(len(self.offsprings) * 0.9)]
            worst_population = sorted(range(len(self.evaluation)),
                                      key=lambda i: self.evaluation[i],
                                      reverse=False)[:int(len(self.offsprings) * 1)]
            # print(self.evaluation[worst_population[0]])
            # print(self.evaluation[best_population[0]])
            # print(self.evaluation[best_offsprings[0]])

            # self.population[:100] = best_population[:]
            # self.population[101:] = best_offsprings[:]

            for i, (best_pop, best_off) in enumerate(zip(best_population, best_offsprings)):
                self.population[worst_population[i]] = self.population[best_pop]
                self.population[worst_population[i+1]] = self.offsprings[best_off]


        # print(best_current_population, "\n", best_offsprings, "\n", worst_population)
        # print(len(best_current_population), (len(best_offsprings)), len(worst_population))


def genetic_algorithm(data, route_length=24, pop_size=1000, eliteism=False):
    """
    Create an instance of a population and perform a genetic mutation algorithm
    to find the best solution to TSP.
    :param data:
    :param route_length:
    :param pop_size:
    :return:
    """
    routes = [r.create_random_route(route_length) for route in range(pop_size)]
    routes = Population(data, routes, eliteism=False)
    results = []

    # Start mutating
    for generation in range(300):
        # Obtain result
        results.append(100000-max(routes.evaluation)) # Best fitness

        # Evolve population
        new_population, generation = routes.evolve()
        routes = Population(data, new_population, generation, eliteism=True)

    # Obtain last result
    # results.append([100000 - max(routes.evaluation),  # Best fitness
    #                 routes.population[routes.evaluation.index(max(routes.evaluation))],  # Best route
    #                 100000 - statistics.mean(routes.evaluation),  # Average fitness
    #                 100000 - statistics.stdev(routes.evaluation)])  # Standard deviation for fitness

    return results

def one_swap_crossover(route):
    """
    Swaps two random alleles with each other
    :param route: The individual to perform crossover
    :return: Mutated individual
    """
    # Sample two random alleles and swap them
    for swap in range(1000):
        seq_idx = list(range(len(route)))
        a1, a2 = random.sample(seq_idx[1:-1], 2)
        new_route = route[:]
        new_route[a1], new_route[a2] = new_route[a2], new_route[a1]
        yield new_route

def one_swap_crossover_system(route):
    # TODO, fjern denne, den brukes ikke.
    """
    Generates a sequence of random swaps
    :param route: The individual to perform crossover
    :return: Mutated individual
    """
    # Create a random index swap
    ind1 = list(range(1, len(route)-1))
    ind2 = list(range(1, len(route)-1))
    random.shuffle(ind1)
    random.shuffle(ind2)
    # Loop through cities and swap
    for city1 in ind1:
        for city2 in ind2:
            swapped_route = route[:]
            swapped_route[city1], swapped_route[city2]\
                = swapped_route[city2], swapped_route[city1]
            assert swapped_route[0] == swapped_route[-1], "start and home is not the same"
            yield swapped_route

def hill_climber(data, route_length=24, first_ten=False):
    """
    Hill climber algorithm that will check a neighboring solution
    If the neighbor solution is better, this becomes the new solution
    if not, keep the old one.
    :param data: Data from CSV file
    :param route_length: Length of routes
    :return:
    """
    # Set up start route
    if first_ten:
        route = r.create_random_route_from_first_n_cities(route_length=10)
    else:
        route = r.create_random_route(route_length)  # Set up a route with 24 cities
    travel_distance = r.get_total_distance(data, route)  # Initiate start solution
    num_evaluations = 1

    # Start hill climbing
    while num_evaluations < 10000:
        move = False
        # Start swapping cities in the route
        for next_route in one_swap_crossover(route):
            updated_dist = r.get_total_distance(data, next_route)
            num_evaluations += 1
            # Climb if better than previous route
            if updated_dist < travel_distance:
                route = next_route
                travel_distance = updated_dist
                move = True
                break
        if not move:
            break
    return travel_distance, route

def exhaustive_search(data, route_length=6):
    """
    Function that searches every possible solution and returns global minimum
    :param route_distance: Function
    :param data: The data that is needed for some functions
    :return: Returns y and x value
    """
    # Setup route permutations
    routes = r.create_permutation_of_routes(route_length)
    fitness = r.get_total_distance(data, routes[0])  # Arbitrary start value
    # Loop through all possible solutions and pick the best one

    for step in routes:
        new_value = r.get_total_distance(data, step)
        if new_value < fitness:
            fitness = new_value
            x_value = step

    return fitness, x_value
