"""
Author - Torstein Gombos
Created - 06.09.2018

Module that contains all the search algorithms used for this assignment

"""
import random
import numpy as np
import routes as r


class Population:
    """
    A class representing a populations of solutions and their
    mutations and crossovers over generations

    Attributes:
        pmx_prob            - Chance for pmx to happen for each parent
        mutation_prob       - Chance for mutation
        data                - Data from CSV
        hybrid              - If hybrid mode is oon
        hybrid_type         - What learning model is used
        population          - All the current solutions to problem
        evaluation          - The fitness of the solutions
        parents             - The selected parents from a population
        offspring           - The resulting children from pmx
        optimized offspring - Offspring after local search
        evaluated_offspring - Fitness of offspring

    """
    def __init__(self, data,  population, hybrid=False, hybrid_type="lamarckian"):
        """
        Initiate hyper parameters, population and evaluate fitness
        :param data: The data from the CSV file
        :param population: All the individuals/routes
        :param hybrid: Boolean. If hybrid is turned on
        :param hybrid_type String. what learning model is used
        """

        # Parameters and data
        self.pmx_prob = 0.8  # 80% chance for pmx crossover in offspring
        self.mutation_prob = 1/len(population[0])  # 1/number_of_cities
        self.data = data  # TSP matrix
        self.hybrid = hybrid  # Checks whether hybrid local search is activated
        self.hybrid_type = hybrid_type  # Either Lamarckian or Baldwinian

        # Set up population and evaluate
        self.population = population  # Initialize population
        self.evaluation = self.evaluate_population(self.population)  # Evaluate current population

        # Evolve new population. Initiated in evolve method
        self.parents = None
        self.offsprings = None
        self.optimized_offspring = None
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

    def evolve(self):
        """
        Evolves the population
        Ends by replacing the population with the new
        one.
        :return:
        """
        # Evolve population
        self.parents = self.select_parents()  # Parents selection
        self.offsprings = self.pmx(self.pmx_prob)  # Create offsprings through pmx crossover
        self.offsprings = self.mutate_population(self.offsprings, self.mutation_prob)  # Mutates the new population
        self.replace_population()

    def select_parents(self):
        """
        Selects parents based on fitness proportionate selection.
        Performs a windowing on fitness beforehand to increase
        the chance for better solutions to be picked. If there
        is to little diversity in the population, population is forced
        to evolve
        :return: parents
        """
        # Perform windowing on fitness
        window_fitness = [(self.evaluation[i] - min(self.evaluation)) for i, item in enumerate(self.evaluation)]

        # If there is too little diversity in population, mutate population
        while sum(window_fitness) == 0:
            self.population = self.mutate_population(self.population, 1)
            self.evaluation = self.evaluate_population(self.population)
            window_fitness = [(self.evaluation[i] - min(self.evaluation)) for i, item in enumerate(self.evaluation)]

        # Create selection wheel and select parents from it
        selection_wheel = [(fitness / sum(window_fitness)) for fitness in window_fitness]
        selection = np.random.choice(range(len(self.population)), len(selection_wheel),
                                     p=selection_wheel)  # Random selection
        parents = [self.population[parent] for parent in selection]
        return parents

    def pmx(self, prob=0.8):
        """
        Performs pmx on parents to create offspring
        Will only perform on 80% on population based on
        a probability.
        :return:
        """
        offsprings = []
        # Loop through mating pool
        for i in range(0, len(self.parents), 2):

            # Probability that pmx don't happen
            if random.random() > prob:
                # Bring parent to next generation
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
                    # Parents are switched to create second offspring
                    parent2, parent1 = self.parents[i+1], self.parents[i]
                    sub_segment1, sub_segment2 = parent2[start:stop], parent1[start:stop]
                offspring = [None] * (len(self.parents[i]))
                offspring[start:stop] = sub_segment1

                # Check if first element in sub_segment2 is in offspring1
                for element in sub_segment2:

                    # Copy elements from sub_segment2 to offspring1
                    if element not in offspring:

                        idx = parent2.index(element)  # Find position for element in p2

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

    def local_search(self, population):
        """
        Performs a hill climb on the offsprings for a local.
        Same algorithm as for hill climber function. Only tweaked to fit this
        problem. Only performs max 20 climbs
        :return:
        """
        optimized_offspring = []
        for individual in population:
            ind = individual.copy()
            travel_distance = r.get_total_distance(self.data, ind)  # Initiate start solution
            num_evaluations = 1

            # Start hill climbing
            # Only perform it a number of iterations. 15 in this case.
            while num_evaluations < 10:
                move = False

                # Start swapping cities in the route
                for next_route in one_swap_crossover_system(ind):
                    updated_dist = r.get_total_distance(self.data, next_route)
                    num_evaluations += 1
                    # Climb if better than previous route
                    if updated_dist < travel_distance:
                        ind = next_route
                        travel_distance = updated_dist
                        move = True
                        break

                if not move:
                    break
            optimized_offspring.append(ind)
        return optimized_offspring

    def mutate_population(self, population, mutation_prob=1/24):
        """
        A probability that an individual will mutate with a swap permutation
        Takes in an arbitrary population so that mutation can be applied to both
        parent or offspring if desired.
        :return:
        """
        offspring = []
        for i, individual in enumerate(population):
            if random.random() < mutation_prob:
                seq_idx = list(range(len(individual)))
                a1, a2 = random.sample(seq_idx[1:-1], 2)
                individual[a1], individual[a2] = individual[a2], individual[a1]

                # Updates offspring
                offspring.append(individual)
            else:
                offspring.append(individual)
        return offspring

    def replace_population(self):
        """
        Method that replaces and evaluates new population.

        Will do so either with a hybrid mode that executes a local search,
        or normal GA with no local search
        :return:
        """
        # Replace and evaluate population
        # Use lamarckian learning model
        if self.hybrid and self.hybrid_type == "lamarckian":  # Perform local search
            self.optimized_offspring = self.local_search(self.offsprings)
            self.evaluated_offspring = self.evaluate_population(self.optimized_offspring)  # acquire fitness

            self.population = self.optimized_offspring  # Replace with optimized offspring
            self.evaluation = self.evaluated_offspring  # Evaluate the new population

        # Use Baldwinian learning model
        elif self.hybrid and self.hybrid_type == "baldwinian":
            self.optimized_offspring = self.local_search(self.offsprings)  # Perform local search
            self.evaluated_offspring = self.evaluate_population(self.optimized_offspring)  # Acquire fitness

            self.population = self.offsprings  # Survivor selection
            self.evaluation = self.evaluated_offspring  # Replace ordinary offspring, keep fitness

        # Hybrid is not selected. Normal survivor selection
        else:
            self.population = self.offsprings  # Replace entire population with offspring (No hybrid)
            self.evaluation = self.evaluate_population(self.population)  # Re-evaluate fitness


def genetic_algorithm(data, route_length=24,
                      pop_size=1000,
                      hybrid=False,
                      generations=150,
                      hybrid_type="lamarckian"):
    """
    Create an instance of a population and perform a genetic mutation algorithm
    to find the best solution to TSP.
    :param hybrid_type: String with the learning model to use if hybrid is on
    :param generations: How many iterations to evolve on
    :param data: Data from CSV file
    :param route_length: Length of each route
    :param pop_size: Size of population
    :param hybrid: Boolean to check if hybrid is activated
    :return: Results from algorithm
    """
    # Create routes
    # Used to create random starting tour
    seed_for_pop = random.random() # Use same seed to initiate the population
    routes = [r.create_random_route(route_length, seed_for_pop) for route in range(pop_size)]

    # Initiate class
    routes = Population(data, routes, hybrid=hybrid, hybrid_type=hybrid_type)

    best_fitness = []  # List that will contain the best fitness of each generation
    # Start evolving over generations
    for generation in range(generations):
        best_fitness.append(100000-max(routes.evaluation))

        # Evolve population
        routes.evolve()  # Initiates evolve method in Population class
    final_fitness = best_fitness[-1]
    return best_fitness, final_fitness, routes.population, routes.evaluation


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
    :param route_length: Length of route
    :param data: The data that is needed for some functions
    :return: Returns fitness and x value
    """
    # Setup route permutations
    routes = r.create_permutation_of_routes(route_length)
    fitness = r.get_total_distance(data, routes[0])  # Arbitrary start value
    # Loop through all possible solutions and pick the best one

    for step in routes:
        new_value = r.get_total_distance(data, step)
        if new_value < fitness:
            fitness = new_value
            route = step

    return fitness, route


def one_swap_crossover(route):
    """
    Swaps two random alleles with each other
    :param route: The individual to perform crossover
    :return: Mutated individual
    """
    # Sample two random cities and swap them
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
