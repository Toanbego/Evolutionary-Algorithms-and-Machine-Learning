"""
Author - Torstein Gombos
Created - 06.09.2018

Module with some simple search algorithms that can be used for optimization.
"""
import random
import ast
import numpy as np
from oblig1 import routes as r
import statistics


class Population:
    """
    A class representing a populations of solutions and their
    mutations and crossovers over generations
    """
    def __init__(self, data,  population, generation=0):
        """
        Take in a single individual and initiate a start population
        :param individual:
        """
        self.data = data  # TSP matrix
        self.generation = generation
        self.population = population  # Initialize population
        self.evaluation = self.evaluate_population()  # Evaluate population
        self.parents = self.select_parents()  # select parents
        self.offsprings = self.pmx()  # Create offsprings from parents

        self.offsprings = self.mutate_offspring()

    def evaluate_population(self):
        """
        Evaluate population
        :return:
        """
        evals = []
        for population in self.population:
            # -100000 is a trick to revert the lowest km to be best fitness
            dist = 100000 - r.get_total_distance(self.data, population)
            evals.append(dist)

        return evals

    def select_parents(self):
        """
        Selects parents based on fitness proportionate selection
        :return:
        """
        p_fps = [(eval/(sum(self.evaluation))) for eval in self.evaluation]
        string_pop = []
        # Convert to string so that np.choice works as 1-d array
        for ind in self.population:
            string_pop.append(str(ind))
        selection = np.random.choice(string_pop, len(p_fps), p=p_fps)
        parents = []
        for parent in selection:
            parents.append(ast.literal_eval(parent))
        return parents

    def pmx(self, prob=0.8):
        """
        Perform pmx on 80% % of parents
        :return:
        """
        # Loop through mating pool
        offsprings = []
        for i in range(0, len(self.parents), 2):
            if random.random() < prob:
                offsprings.append(self.parents[i])
                offsprings.append(self.parents[i+1])
                continue
            try:
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
            except IndexError:
                break
        return offsprings

    def mutate_offspring(self, prob=0.7):
        """
        A probability that an offspring will mutate with a swap
        :return:
        """

        try:
            for i, offspring in enumerate(self.offsprings):
                if random.random() < prob:
                    seq_idx = list(range(len(offspring)))
                    a1, a2 = random.sample(seq_idx[1:-1], 2)
                    offspring[a1], offspring[a2] = offspring[a2], offspring[a1]
                    self.offsprings[i] = offspring
                else:
                    continue

        except TypeError:
            print("stop")




def genetic_algorithm(data, route_length=24, pop_size=10):
    """
    Create an instance of a population and perform a genetic mutation algorithm
    to find the best solution to TSP.
    :param data:
    :param route_length:
    :param pop_size:
    :return:
    """
    routes = [r.create_random_route(route_length) for route in range(pop_size)]
    routes = Population(data, routes)
    routes.mutate_offspring()
    while routes.generation < 100:
        pass

    # for i in range(routes.population):


    return 1, 1

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
        copy = route[:]
        copy[a1], copy[a2] = copy[a2], copy[a1]
        yield copy

def one_swap_crossover_system(route):
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

def hill_climber(data, route_length=24, num_of_rand_resets=100):
    """
    Hill climber algorithm that will check a neighboring solution
    If the neighbor solution is better, this becomes the new solution
    if not, keep the old one.
    :param data:
    :param route_length:
    :param num_of_rand_resets:
    :return:
    """
    route = r.create_random_route(route_length)  # Set up a route with 24 cities
    travel_distance = r.get_total_distance(data, route)  # Initiate start solution
    num_evaluations = 1
    while num_evaluations < 10000:
        move = False
        # Start swapping cities in the route
        for next_route in one_swap_crossover(route):
            updated_dist = r.get_total_distance(data, next_route)
            num_evaluations += 1
            # Climb if better
            if updated_dist < travel_distance:
                route = next_route
                travel_distance = updated_dist
                move = True
                break
        if not move:
            break
    return travel_distance, route

def exhaustive_search(route_distance, data, route_length=6):
    """
    Function that searches every possible solution and returns global minimum
    :param route_distance: Function
    :param data: The data that is needed for some functions
    :return: Returns y and x value
    """
    # Setup route permutations
    routes = r.create_permutation_of_routes(route_length)
    fitness = route_distance(data, routes[0])  # Arbitrary start value
    # Loop through all possible solutions and pick the best one
    for step in routes:
        new_value = route_distance(data, step)
        if new_value < fitness:
            fitness = new_value
            x_value = step
    return fitness, x_value
