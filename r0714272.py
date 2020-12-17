from math import isinf

import Reporter
import numpy as np
import random as rnd
import heapq as hq
import matplotlib.pyplot as plt

class r0714272:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):

        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()
        seed = rnd.randrange(0, 1239719)
        # best seed: 683676
        print("SEED = {}".format(12345678))
        rnd.seed(12345678)
        np.random.seed(12345678)

        island_params = [IslandParameters(mu=200, epsilon=5, q=10, min_dist=3, k=3)] * 4
        params = AlgorithmParameters(la=200, init_alpha=0.1, init_beta=0.75, island_params=island_params, exchange_size=10, exchange_rate=50)
        tsp = TSP(distance_matrix,params)

        # Initialize the population
        tsp.initialize()
        best_sol = None
        best_obj = None
        while tsp.should_continue():
            (mean_obj, best_obj, best_sol) = tsp.report_values()
            time_left = self.reporter.report(mean_obj, best_obj, best_sol)
            if time_left < 0:
                print("Ran out of time!")
                break

            tsp.update()

        real_best = fitness(best_sol)
        if real_best != best_obj:
            raise RuntimeError("WRONG SOLUTION: {} != {}".format(real_best, best_obj))

        return 0


class TSP:
    """ Class that represents an evolutionary algorithm to solve the Travelling Salesman Problem. """
    distance_matrix = None
    n = None
    def __init__(self, distance_matrix, params):
        TSP.distance_matrix = distance_matrix
        TSP.n = distance_matrix.shape[0]
        self.params = params
        self.islands = []
        self.sc = SlopeChecker(-0.0000000001, 0.5)
        self.iteration = 0

    def initialize(self):
        """
        Initialize the population using random permutations and the initial values specified in the AlgorithmParameters object.
        """
        permutations = self.get_initial_permutations()
        population = [Individual(permutations[j], fitness(permutations[j]), 2 * (rnd.random() - 0.5) * 0.02 + self.params.init_alpha, self.params.init_beta) for j in range(self.params.la)]
        island_la = int(self.params.la / len(self.params.island_params))
        for island_params in self.params.island_params:
            idx = rnd.randrange(0, len(population))
            leader = population.pop(idx)
            #population.sort(key=lambda ind: distance(ind.perm, leader.perm), reverse=True)
            island_pop = [population.pop(0) for _ in range(island_la - 1)]
            island_pop.append(leader)
            self.islands.append(Island(island_pop, island_params))

    def get_initial_permutations(self):
        permutations = []
        for _ in range(self.params.la):
            start = rnd.randrange(0, TSP.n)
            available_cities = set(range(TSP.n))
            perm = np.zeros(TSP.n, dtype=int)
            perm[0] = start
            available_cities.remove(start)
            for i in range(1, TSP.n):
                perm[i] = min(available_cities, key=lambda city: self.distance_matrix[perm[i - 1]][city])
                available_cities.remove(perm[i])
            (i, j) = random_ind(TSP.n)
            flip(perm, i, j)
            permutations.append(perm)
        return permutations

    def distribute_population(self):
        for i in range(len(self.islands)):
            elite_victims = rnd.sample(self.islands[i].elites, k=min(len(self.islands[i].elites), 2))
            peasants_victims = rnd.sample(self.islands[i].peasants, k=self.params.exchange_size - 2)
            self.islands[(i+1) % len(self.islands)].peasants += peasants_victims
            self.islands[(i + 1) % len(self.islands)].elites += elite_victims
            #self.islands[i].redivide()


    def update(self):
        self.iteration += 1
        print("----------------{}---------------------".format(self.iteration))
        best_obj = min([island.best_individual.fitness for island in self.islands])
        self.sc.update(best_obj)

        for island in self.islands:
            #print("best: {} \t mean: {} \t worst: {}".format(island.best_individual.fitness, island.mean_fitness, island.worst_fitness))
            island.update()

        best_ind = min([island.best_individual for island in self.islands], key=lambda ind: ind.fitness)
        print("best: {}, mean: {}, alpha: {}, beta: {}".format(best_ind.fitness, self.report_values()[0], best_ind.alpha, best_ind.beta))

        if self.iteration % self.params.exchange_rate == 0:
            print("Redistributing population...")
            #self.distribute_population()

    def report_values(self):
        best = min([island.best_individual for island in self.islands], key=lambda ind: ind.fitness)
        return sum(island.mean_fitness for island in self.islands) / 4.0, best.fitness, best.perm

    def should_continue(self):
        return self.sc.should_continue()

class Island:
    """ A class that represents an island of an the evolutionary algorithm. """

    def __init__(self, population, params):
        """
        Create a new Island object.
        """
        self.params = params
        (self.elites, self.peasants) = divide(population, params.epsilon, params.min_dist)
        self.offsprings = []
        self.la = len(population)
        self.best_individual = None
        self.mean_fitness = 0
        self.worst_fitness = 0
        self.distribution = [0.2, 0.3]
        self.update_statistics()
        self.ai_ls = 0
        self.lsc = 0
        self.ai_mut = 0
        self.mutc = 0

    def create_offsprings(self):

        #elite_pool = [(elite1, elite2) for elite1 in self.elites for elite2 in self.elites]
        elite_pool = [(select(self.elites, self.params.k), select(self.elites, self.params.k)) for _ in range(int(self.distribution[0] * self.params.mu))]
        self.do_crossovers(elite_pool)

        elite_peasants_pool = [(select(self.elites, self.params.k), select(self.peasants, self.params.k)) for _ in range(int(self.distribution[1] * self.params.mu))]
        self.do_crossovers(elite_peasants_pool)

        peasants_pool = [(select(self.peasants, self.params.k), select(self.peasants, self.params.k)) for _ in range(self.params.mu - len(elite_pool) - len(elite_peasants_pool))]
        self.do_crossovers(peasants_pool)


    def do_crossovers(self, mating_pool):
        for (parent1, parent2) in mating_pool:
            chance = rnd.random()
            parent1_ready = chance <= parent1.beta
            parent2_ready = chance <= parent2.beta

            if parent1_ready and parent2_ready and parent1 != parent2:
                child = self.create_offspring(parent1, parent2)
                self.offsprings.append(child)
                parent1.beta = min(1.0, max(0.0, child.fitness / parent1.fitness * parent1.beta))
                parent2.beta = min(1.0, max(0.0, child.fitness / parent2.fitness * parent2.beta))
            elif parent1_ready:
                child = self.create_offspring(parent1)
                self.offsprings.append(child)
                parent1.beta = min(1.0, max(0.0, child.fitness / parent1.fitness * parent1.beta))
            elif parent2_ready:
                child = self.create_offspring(parent2)
                self.offsprings.append(child)
                parent2.beta = min(1.0, max(0.0, child.fitness / parent2.fitness * parent2.beta))

    def create_offspring(self, parent1, parent2=None):
        if parent2 is None:
            perm_offspring = np.copy(parent1.perm)
            mutate(perm_offspring, perm_offspring.shape[0]-1) #TODO: what should max_length be?
            new_fitness = fitness(perm_offspring)
            new_beta = min(1.0, max(0.0, new_fitness / parent1.fitness * parent1.beta))
            return Individual(perm_offspring, new_fitness, parent1.alpha, new_beta)
        else:
            perm_offspring = recombine(parent1.next_cities, parent2.next_cities)
            new_alpha = min(1.0, max(0.0, parent1.alpha + 2 * (rnd.random() - 0.5) * (parent2.alpha - parent1.alpha)))
            new_beta = min(1.0, max(0.0, parent1.beta + 2 * (rnd.random() - 0.5) * (parent2.beta - parent1.beta)))
            return Individual(perm_offspring, fitness(perm_offspring), new_alpha, new_beta)


    def mutate(self):
        """
        Mutate the population.
        """
        for ind in self.offsprings + self.peasants:
            if rnd.random() < ind.alpha:
                self.mutc += 1
                mutate(ind.perm, ind.perm.shape[0]-1) #TODO: what should max_length be?
                self.ai_mut -= ind.fitness
                ind.fitness = fitness(ind.perm)
                self.ai_mut += ind.fitness
                ind.optimized = False

    def elimination(self):
        """
        Eliminate certain Individuals from the population.
        """
        o_elites, o_peasants = divide(self.offsprings, self.params.epsilon, self.params.min_dist)
        new_population = eliminate(self.elites, self.peasants, o_elites, o_peasants, self.params.q, self.params.epsilon, self.la)
        self.elites, self.peasants = divide(new_population, self.params.epsilon, self.params.min_dist)
        self.offsprings = []

    def redivide(self):
        self.elites, self.peasants = divide(self.elites + self.peasants, self.params.epsilon, self.params.min_dist)

    def local_search(self):
        """
        Apply local search to optimize the population.
        """
        #for ind in self.elites:
        #    #iterative_search(ind, 2)
        #    new_perm, new_fitness = deep_search(ind.perm, ind.fitness, ind.k, 2, 50)
        #    if not new_perm is None:
        #        ind.perm = new_perm
        #        ind.fitness = new_fitness
        #    else:
        #        ind.k += 1

        for ind in self.elites: # + self.peasants:
            if not ind.optimized:
                self.lsc += 1
                self.ai_ls -= ind.fitness
                new_fitness = local_search(ind.perm)
                ind.fitness = new_fitness

                new_perm,new_fitness = local_search2(ind.perm, ind.fitness, ind.k)
                if not new_perm is None:
                    ind.optimized = False
                    ind.perm = new_perm
                    ind.fitness = new_fitness
                else:
                    ind.k += +1
                    if ind.k >= TSP.n:
                        ind.optimized = True
                        raise RuntimeError("LOCAL OPTIMA")
                        #print("Found local optima")
                self.ai_ls += ind.fitness


    def update_statistics(self):
        """
        Return a tuple containing the following:
            - the mean objective function value of the population
            - the best objective function value of the population
            - a 1D numpy array in the cycle notation containing the best solution
              with city numbering starting from 0
        :return: A tuple (m, bo, bs) that represent the mean objective, best objective and best solution respectively
        """
        mean = 0
        best_fitness = float('inf')
        worst_fitness = -1
        best_individual = None
        for ind in self.elites + self.peasants:
            f = ind.fitness
            mean += f
            if f < best_fitness:
                best_fitness = f
                best_individual = ind
            if f > worst_fitness:
                worst_fitness = f
        mean = mean / (len(self.elites) + len(self.peasants))

        self.best_individual = best_individual
        self.worst_fitness = worst_fitness
        self.mean_fitness = mean

    def update(self):
        """
        Update the population.
        """
        self.create_offsprings()
        self.mutate()
        self.local_search()
        self.elimination()
        self.update_statistics()
        print("elites: {}, peasants: {}, mutation improvement: {} ({}), local search improvement {} ({})" \
              .format(len(self.elites), len(self.peasants), self.ai_mut, self.mutc, self.ai_ls, self.lsc))
        self.ai_mut = 0
        self.ai_ls = 0
        self.mutc = 0
        self.lsc = 0

class Individual:
    """
    A class that represents an order in which to visit the cities.
    This class will represent an individual in the population.
    """

    def __init__(self, perm, f, alpha, beta):
        """
        Create a new TSPTour with the specified parameters
        :param perm: The permutation, this should be a numpy.array containing the order of the cities.
        :param fitness: The fitness value for this individual.
        """
        self.perm = perm
        self.fitness = f
        self.next_cities = np.zeros(TSP.n, dtype=int)
        for i in range(TSP.n):
            self.next_cities[self.perm[i]] = self.perm[(i+1) % TSP.n]
        self.alpha = alpha
        self.beta = beta
        self.k = 2
        self.optimized = False

    def get_next(self, city):
        """
        Get the city that follows next to the given city
        :param city: The number of the city (starting from zero)
        :return: The number of the city that follows the given city in this Individual.
        """
        return self.next_cities[city]

class SlopeChecker:
    """ A class for checking if the slope is still steep enough. """

    def __init__(self, max_slope, weight):
        """
        Create a new SlopeChecker.
        :param max_slope: The minimal slope that the convergence graph should have.
        """
        self.max_slope = max_slope
        self.min_slope = float('inf')
        self.best_objs = []
        self.slope = 0
        self.weight = weight

    def update(self, best_obj):
        """
        Update this SlopeChecker
        :param best_obj: The current best objective
        """
        self.best_objs.append(best_obj)
        if len(self.best_objs) >= 2:
            new_slope = self.best_objs[-1] - self.best_objs[-2]
            if new_slope < self.min_slope:
                self.min_slope = new_slope
            self.slope = (1 - self.weight) * self.slope + self.weight * new_slope

    def get_slope_progress(self):
        """
        Get a value between 0 and 1 indicating how far the slope is from its target.
        :return: A value between 0 and 1 indicating how far the slope is from its target.
        """
        if len(self.best_objs) < 2:
            return 0
        return 1 - abs(self.max_slope - self.slope) / abs(self.max_slope - self.min_slope)

    def should_continue(self):
        """
        Check if the algorithm shoud continue.
        :return: True if the algorithm should continue, False otherwise.
        """
        return self.slope <= self.max_slope or self.slope == 0

class AlgorithmParameters:
    """
    A class that contains all the information to run the genetic algorithm.
    """

    def __init__(self, la, init_alpha, init_beta, island_params, exchange_size, exchange_rate):
        """
        Create a new AlgorithmParameters object.
        :param la: The population size
        :param mu: The amount of tries to create offsprings (not every try will result in offsprings)
        :param beta: The probability that two parents
        """
        self.la = la
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.island_params = island_params
        self.exchange_size = exchange_size
        self.exchange_rate = exchange_rate

class IslandParameters:
    def __init__(self, mu, epsilon, q, min_dist, k):
        self.mu = mu
        self.epsilon = epsilon
        self.q = q
        self.min_dist = min_dist
        self.k = k

def select(population,k):
    return min(rnd.choices(population, k=k), key=lambda ind: ind.fitness)

def eliminate(elites, peasants, o_elites, o_peasants, q, epsilon, la):
    return pick_survivors(elites + o_elites + peasants + o_peasants, la, q) #, pick_survivors(peasants + o_peasants, la - epsilon, q)

def pick_survivors(population, la, q):
    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    survivors = []

    while len(survivors) < la:
        survivor = sorted_pop.pop(0)
        survivors.append(survivor)
        victim = min(rnd.choices(sorted_pop, k=min(q, len(sorted_pop))), key=lambda ind: ind.fitness)
        sorted_pop.remove(victim)

    return survivors

def calc_fitness_swap(perm, f, i, j):
    if i == j:
        return f

    i, j = min(i, j), max(i, j)
    if i == 0 and j == TSP.n-1:
        i,j = j,i

    f -= TSP.distance_matrix[perm[i]][perm[(i + 1) % TSP.n]]
    f -= TSP.distance_matrix[perm[(i - 1) % TSP.n]][perm[i]]
    f -= TSP.distance_matrix[perm[j]][perm[(j + 1) % TSP.n]]
    if j == (i+1) % TSP.n or (j+1) % TSP.n == i:
        f += TSP.distance_matrix[perm[j]][perm[i]]
    else:
        f -= TSP.distance_matrix[perm[(j - 1) % TSP.n]][perm[j]]
    f += TSP.distance_matrix[perm[(i - 1) % TSP.n]][perm[j]]
    f += TSP.distance_matrix[perm[j]][perm[(i + 1) % TSP.n]]
    f += TSP.distance_matrix[perm[(j - 1) % TSP.n]][perm[i]]
    f += TSP.distance_matrix[perm[i]][perm[(j + 1) % TSP.n]]
    return f

def local_search(perm):
    old_f = fitness(perm)
    worst_edge = max(range(TSP.n), key=lambda i: TSP.distance_matrix[perm[i]][perm[(i+1) % TSP.n]])
    best_edge = min([i for i in range(TSP.n)], key= lambda i:calc_fitness_swap(perm, old_f, worst_edge, i))
    new_f = calc_fitness_swap(perm, old_f, worst_edge, best_edge)
    perm[worst_edge], perm[best_edge] = perm[best_edge], perm[worst_edge]
    f = fitness(perm)

    if round(f,4) != round(new_f,4):
        raise RuntimeError("[{}, {}]: {} != {}".format(worst_edge, best_edge, f, new_f))
    return f

def calc_gain(perm, i, j):
    n = perm.shape[0]
    gain = 0
    if (i+1) % n != j:
        gain -= TSP.distance_matrix[perm[(i-1) % TSP.n]][perm[i]]
        gain -= TSP.distance_matrix[perm[i]][perm[(i+1) % TSP.n]]
        gain -= TSP.distance_matrix[perm[(j-1) % TSP.n]][perm[j]]
        gain += TSP.distance_matrix[perm[i]][perm[j]]
        gain += TSP.distance_matrix[perm[(j-1) % TSP.n]][perm[i]]
        gain += TSP.distance_matrix[perm[(i-1) % TSP.n]][perm[(i+1) % TSP.n]]
    return gain

def local_search2(perm, old_fitness, k):
    """
    Search for a neighbouring permutation that has a better fitness.
    :param perm: The permutation to improve.
    :param old_fitness: The fitness to improve.
    :param k: The maximum length of the inversions.
    :return: A tuple (p,f) where p is new permutation that has better fitness than the given one or None if no such permutation can be found and f its fitness.
    """
    inversions = [(i, (i+k) % TSP.n) for i in range(TSP.n)]
    best_inversion = None
    best_fitness = old_fitness
    for i,j in inversions:
        new_fitness = estimate_fitness(perm, old_fitness, i, j)
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            best_inversion = (i,j)

    if not best_inversion is None and best_inversion[0] != best_inversion[1]:
        flip(perm, best_inversion[0], best_inversion[1])
        f = fitness(perm)
        if round(best_fitness,4) != round(f, 4):
            print(best_inversion)
            raise RuntimeError("{} != {}".format(best_fitness, f))
        return perm, fitness(perm)

    return None, None

def fitness(perm):
    """
    Calculate the fitness value of the given tour.
    :param perm: The order of the cities.
    :return: The fitness value of the given tour.
    """
    return np.sum(np.array([TSP.distance_matrix[perm[i % TSP.n]][perm[(i + 1) % TSP.n]] for i in range(TSP.n)]))

def estimate_fitness(perm, old_fitness, i, j):
    new_fitness = old_fitness
    k = i
    while k % TSP.n != (j - 1) % TSP.n:
        new_fitness -= TSP.distance_matrix[perm[k % TSP.n]][perm[(k + 1) % TSP.n]]
        new_fitness += TSP.distance_matrix[perm[(k + 1) % TSP.n]][perm[k % TSP.n]]
        k += 1
    new_fitness -= TSP.distance_matrix[perm[(i - 1) % TSP.n]][perm[i]]
    new_fitness -= TSP.distance_matrix[perm[(j - 1) % TSP.n]][perm[j]]
    new_fitness += TSP.distance_matrix[perm[(i - 1) % TSP.n]][perm[(j - 1) % TSP.n]]
    new_fitness += TSP.distance_matrix[perm[i]][perm[j]]

    return new_fitness

def divide(population, elite_size, min_distance):
    """
    Divide the given population in elites and peasants.
    :param population: The population of Individuals to divide.
    :param elite_size: The size of the elites.
    :param min_distance: The minimal distance the members of the elite group should have
    :return: a tuple (elites,peasants).
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    elites = []
    k = 0
    while len(elites) < elite_size and k < len(sorted_pop):
        allowed = True
        for elite in elites:
            if distance(sorted_pop[k].perm, elite.perm) < min_distance:
                allowed = False
                break
        if allowed:
            elites.append(sorted_pop.pop(k))
        else:
            k += 1

    return elites, sorted_pop

def recombine(next_cities1, next_cities2):
    """
    :param next_cities1: A numpy array a where a[perm1[i]] = perm1[i+1 % n]
    :param next_cities2: A numpy array a where a[perm2[i]] = perm2[i+1 % n]
    :return: A new permutation that resembles an offspring.
    """
    perm_offspring = np.zeros(shape=TSP.n, dtype=int)
    perm_offspring_lu = set()
    start = rnd.randrange(0, TSP.n)
    perm_offspring[0] = start
    perm_offspring_lu.add(start)
    available_cities = set(range(TSP.n))
    available_cities.remove(perm_offspring[0])

    for i in range(1, TSP.n):
        c = perm_offspring[i - 1]
        c1 = next_cities1[c]
        c2 = next_cities2[c]

        c1_ok = c1 not in perm_offspring_lu
        c2_ok = c2 not in perm_offspring_lu
        if c1_ok and c2_ok:
            if TSP.distance_matrix[c][c1] < TSP.distance_matrix[c][c2]:
                perm_offspring[i] = c1
            else:
                perm_offspring[i] = c2
        elif c1_ok:
            perm_offspring[i] = c1
        elif c2_ok:
            perm_offspring[i] = c2
        else:
            perm_offspring[i] = rnd.choices(tuple(available_cities), k=1)[0]

        perm_offspring_lu.add(perm_offspring[i])
        available_cities.remove(perm_offspring[i])

    return perm_offspring

def mutate(perm, max_length):
    """
    Perform Inversion Mutation on the given permutation.
    :param perm: The numpy.array to permute.
    :param max_length: The maximum length of the inversion.
    """
    start = rnd.randrange(TSP.n)
    end = (start + rnd.randrange(1, max_length + 1)) % TSP.n
    flip(perm,start,end)

def flip(perm, start, end):
    """
    Flip the given permutation; perm[start] will now be perm[end-1].
    :param perm: The permutation to flip.
    :param start: The index of the first element to flip.
    :param end: The index of the first element not to flip.
    """
    if end > start:
        perm[start:end] = np.flip(perm[start:end])
    else:
        flipped_perm = np.concatenate((np.flip(perm[:end]), np.flip(perm[start:])))
        perm[start:] = flipped_perm[:(perm.shape[0] - start)]
        if end > 0:
            perm[:end] = flipped_perm[-end:]

def distance(perm1, perm2):
    """
    Calculate the distance between two permutations.
    :param perm1: The first permutations.
    :param perm2: The second permutations.
    :return: The distance between the two given permutations.
    """
    start_idx = np.flatnonzero(perm2 == perm1[0])[0]
    if start_idx < TSP.n / 2:
        return min_swap(perm1, np.roll(perm2, -start_idx))
    else:
        return min_swap(perm1, np.roll(perm2, TSP.n - start_idx))
    #return TSP.n - 1 - common_edges(perm1, perm2)

### UTILITY METHODS

def random_ind(n):
    """
    Generate two random numbers between 0 (inclusive) and n (exclusive)
    :param n: The exclusive upperbound
    :return: a tuple (i1,i2) where 0 <= i1,i2 < n and i1 != i2
    """
    i1 = rnd.randrange(0, n)
    i2 = rnd.randrange(0, n)
    while i1 == i2:
        i2 = rnd.randrange(0, n)

    return min(i1, i2), max(i1, i2)

def swap_copy(x, i, j):
    """
    Return a copy of the given numpy array with the ith and jth value swapped.
    :param x: The numpy array to make a copy of and swap ith and jth value.
    :param i: The first index.
    :param j: The second index.
    :return: A copy of the given list with the ith and jth value swapped.
    """
    y = np.array(x)
    y[i],y[j] = y[j], y[i]
    return y

def flip_copy(x, i, j):
    """
    Return a copy of the given numpy array with x[i+1:j] flipped.
    :param x: The numpy array to perform the operation on.
    :param i: The first index.
    :param j: The second index.
    :return: A copy of the given numpy array with x[i+1:j] flipped.
    """
    y = np.array(x)
    y[i+1:j] = np.flip(y[i+1:j])
    return y

def flatten(x):
    """
    Flatten the given list of lists.
    :param x: The list of lists to flatten.
    :return: A flattened list.
    """
    return [a for sublist in x for a in sublist]

def min_swap(a, b):
    """
    Calculate the minimal amount of swaps that are needed to make b the same as a.
    :param a: The first array (np.array)
    :param b: The second array (np.array)
    :return: The minimal amount of swaps that are needed to make b the same as a.
    """
    mp = {}
    for i in range(TSP.n):
        mp[b[i]] = i

    for i in range(TSP.n):
        b[i] = mp[a[i]]

    arrPos = [[0 for x in range(2)] for y in range(TSP.n)]

    for i in range(TSP.n):
        arrPos[i][0] = b[i]
        arrPos[i][1] = i

    arrPos.sort()
    vis = [False] * (TSP.n)

    ans = 0

    for i in range(TSP.n):

        if vis[i] or arrPos[i][1] == i:
            continue

        cycle_size = 0
        j = i

        while not vis[j]:
            vis[j] = 1

            j = arrPos[j][1]
            cycle_size += 1

        ans += (cycle_size - 1)

    return ans

def common_edges(a, b):
    edges = set()
    for i in range(TSP.n):
        edges.add((a[i], a[(i+1) % TSP.n]))
        edges.add((b[i], b[(i+1) % TSP.n]))

    return 2 * TSP.n - len(edges)