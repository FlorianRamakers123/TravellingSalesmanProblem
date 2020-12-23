import Reporter
import numpy as np
import random as rnd

class r0714272:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):

        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        island_params = [IslandParameters(mu=200, epsilon=15, q=30, min_dist=5, k=3, distribution=[0.25, 0.35], parent_reward=1.02)]
        island_params += [IslandParameters(mu=200, epsilon=10, q=30,min_dist=5, k=3, distribution=[0.2, 0.4], parent_reward=1.02)]
        island_params += [IslandParameters(mu=200, epsilon=10, q=30, min_dist=5, k=3, distribution=[0.3, 0.35], parent_reward=1.02)]
        island_params += [IslandParameters(mu=200, epsilon=15, q=30, min_dist=5, k=3, distribution=[0.2, 0.3], parent_reward=1.02)]
        params = AlgorithmParameters(la=200, init_alpha=0.1, var_alpha=0.02, init_beta=0.75, island_params=island_params, exchange_size=(2, 10), exchange_rate=25, max_slope=-0.00000001, slope_weight=0.6)
        tsp = TSP(distance_matrix,params)

        # Initialize the population
        tsp.initialize()

        try:
            while not tsp.should_stop():
                (mean_obj, best_obj, best_sol) = tsp.report_values()
                time_left = self.reporter.report(mean_obj, best_obj, best_sol)
                if time_left < 0:
                    break

                tsp.update()
        except KeyboardInterrupt:
            pass

        return 0

### ------------------- ###
###  ALGORITHM CLASSES  ###
### ------------------- ###

class TSP:
    """ Class for solving the Travelling Salesman Problem. """

    distance_matrix = None          # Static variable for the distance matrix
    weight_matrix = None            # Static variable for the weight matrix
    n = None                        # Static variable for the amount of cities

    def __init__(self, distance_matrix, params):
        """
        Create a new TPS object
        :param distance_matrix: The distance matrix for the TSP problem
        :param params: The AlgorithmParameters object that specifies the parameters for the algorithm and islands
        """
        TSP.distance_matrix = distance_matrix
        TSP.weight_matrix = TSP.calc_weight_matrix(distance_matrix)


        TSP.n = distance_matrix.shape[0]
        self.params = params
        self.islands = []

        self.best_objs = []
        self.slope = 0

        self.iteration = 0

    @staticmethod
    def calc_weight_matrix(distance_matrix):
        s = np.copy(distance_matrix)
        s[np.where(s == 0)] = np.inf
        weight_matrix = 1 / np.power(s,3)
        for row in range(weight_matrix.shape[0]):
            weight_matrix[weight_matrix > 0] = weight_matrix[weight_matrix > 0] / np.sum(weight_matrix[row])

        return weight_matrix

    def initialize(self):
        """
        Initialize the population using random permutations.
        """
        permutations = TSP.generate_start_permutations(20)
        permutations += [np.random.permutation(TSP.n) for _ in range(self.params.la - len(permutations))]
        rnd.shuffle(permutations)

        alphas = [2 * (rnd.random() - 0.5) * self.params.var_alpha + self.params.init_alpha for _ in range(self.params.la)]
        population = [Individual(permutations[j], fitness(permutations[j]), alphas[j], self.params.init_beta) for j in range(self.params.la)]
        island_la = int(self.params.la / len(self.params.island_params))

        for island_params in self.params.island_params:
            island_pop = [population.pop(0) for _ in range(island_la)]
            self.islands.append(Island(island_pop, island_params))

    @staticmethod
    def generate_start_permutations(m):
        """
        Calculate m starting permutations by choosing a random start city and performing a greedy walk.
        :param m: The amount of starting permutations to generate.
        :return:
        """
        permutations = []
        for _ in range(m):
            start = rnd.randrange(0, TSP.n)
            available_cities = list(range(TSP.n))
            perm = np.zeros(TSP.n, dtype=int)
            perm[0] = start
            available_cities.remove(start)
            for i in range(1, TSP.n):
                choices = rnd.choices(available_cities, k=len(available_cities), weights=TSP.weight_matrix[perm[i-1]][available_cities])
                perm[i] = min(choices, key= lambda city: TSP.distance_matrix[perm[i-1]][city])
                available_cities.remove(perm[i])
            (i, j) = random_ind(TSP.n)
            invert(perm, i, j)
            permutations.append(perm)
        return permutations

    def distribute_population(self):
        """
        Exchange some individuals between islands
        """
        for i,island in enumerate(self.islands):
            elite_victims = [select(island.elites, min(len(island.elites), island.params.k)).cpy() for _ in range(self.params.ex_size_elite)]
            peasants_victims = [select(self.islands[i].peasants, island.params.k).cpy() for _ in range(self.params.ex_size_peasants)]
            self.islands[(i+1) % len(self.islands)].peasants += peasants_victims
            self.islands[(i+1) % len(self.islands)].elites += elite_victims

    def update_slope(self, best_obj):
        """
        Update the slope variable
        :param best_obj: The current best objective
        """
        self.best_objs.append(best_obj)
        if len(self.best_objs) >= 2:
            new_slope = self.best_objs[-1] - self.best_objs[-2]
            self.slope = (1 - self.params.slope_weight) * self.slope + self.params.slope_weight * new_slope

    def update(self):
        """
        Update all the islands and the stopping criterion check.
        """
        self.iteration += 1

        for island in self.islands:
            island.update()

        mo, bo, _ = self.report_values()
        print("[{}] best: {} \t\t mean: {} ".format(self.iteration, bo, mo))
        self.update_slope(bo)

        if self.iteration % self.params.exchange_rate == 0:
            self.distribute_population()

    def report_values(self):
        """
        Return a tuple containing the following:
            - the mean fitness of the population
            - the best fitness of the population
            - a 1D numpy array in the cycle notation containing the best solution
              with city numbering starting from 0
        :return: A tuple (m, bo, bs) that represent the mean objective, best objective and best solution respectively
        """
        best = min([island.best_individual for island in self.islands], key=lambda ind: ind.fitness)
        return sum(island.mean_fitness for island in self.islands) / 4.0, best.fitness, best.perm

    def should_stop(self):
        """
        Check if the algorithm should stop.
        :return: True if the algorithm should stop, False otherwise.
        """
        return self.slope >= self.params.max_slope and self.iteration > 2

class Island:
    """ A class that represents an island of an the evolutionary algorithm. """

    def __init__(self, population, params):
        """
        Create a new Island object.
        :param population: The population for this island.
        :param params: The IslandParameters object for this Island.
        """
        self.params = params
        (self.elites, self.peasants) = divide(population, params.epsilon, params.min_dist)
        self.offsprings = []
        self.la = len(population)

        self.best_individual = None
        self.mean_fitness = 0
        self.worst_fitness = 0
        self.update_statistics()        # update_statistics will initialize the above three values


    def create_offsprings(self):
        """
        Create offsprings.
        """
        elite_pool = [(select(self.elites, self.params.k), select(self.elites, self.params.k)) for _ in range(int(self.params.distribution[0] * self.params.mu))]
        self.do_crossovers(elite_pool)

        elite_peasants_pool = [(select(self.elites, self.params.k), select(self.peasants, self.params.k)) for _ in range(int(self.params.distribution[1] * self.params.mu))]
        self.do_crossovers(elite_peasants_pool)

        peasants_pool = [(select(self.peasants, self.params.k), select(self.peasants, self.params.k)) for _ in range(self.params.mu - len(self.offsprings))]
        self.do_crossovers(peasants_pool)

    def do_crossovers(self, mating_pool):
        """
        Perform all crossovers in the given mating pool.
        The offsprings will be stored in self.offsprings.
        :param mating_pool: A list of pairs of parents.
        """
        for (parent1, parent2) in mating_pool:
            chance = rnd.random()
            parent1_ready = chance <= parent1.beta
            parent2_ready = chance <= parent2.beta

            if parent1_ready and parent2_ready and parent1 != parent2:
                child = Island.create_offspring(parent1, parent2)
                self.offsprings.append(child)
                self.evaluate_parent(parent1, child)
                self.evaluate_parent(parent2, child)
            elif parent1_ready:
                child = Island.create_offspring(parent1)
                self.offsprings.append(child)
                self.evaluate_parent(parent1, child)
            elif parent2_ready:
                child = Island.create_offspring(parent2)
                self.offsprings.append(child)
                self.evaluate_parent(parent2, child)


    def evaluate_parent(self, parent, child):
        if child.fitness < parent.fitness:
            parent.beta = min(1.0, parent.beta * self.params.parent_reward)

    @staticmethod
    def create_offspring(parent1, parent2=None):
        """
        Create an offspring using the given parent(s).
        If parent2 == None then the offspring will created with mutation.
        :param parent1: The first parent
        :param parent2: The (optional) second parent
        :return: An offspring of the given parent(s)
        """
        if parent2 is None:
            perm_offspring = np.copy(parent1.perm)
            mutate(perm_offspring, perm_offspring.shape[0]//4)
            new_fitness = fitness(perm_offspring)
            return Individual(perm_offspring, new_fitness, parent1.alpha, parent1.beta)
        else:
            perm_offspring = recombine(parent1.next_cities, parent2.next_cities)
            new_fitness = fitness(perm_offspring)
            new_alpha = min(1.0, max(0.0, parent1.alpha + 2 * (rnd.random() - 0.5) * (parent2.alpha - parent1.alpha)))
            return Individual(perm_offspring, new_fitness, new_alpha, (parent1.beta + parent2.beta) / 2)

    def mutate(self):
        """
        Mutate the population.
        """
        for ind in self.offsprings:
            if rnd.random() < ind.alpha:
                mutate(ind.perm, ind.perm.shape[0]//8)
                ind.fitness = fitness(ind.perm)
                ind.optimized = False

    def elimination(self):
        """
        Eliminate certain Individuals from the population.
        """
        new_population = eliminate(self.elites + self.peasants + self.offsprings, self.la, self.params.q)
        self.elites, self.peasants = divide(new_population, self.params.epsilon, self.params.min_dist)
        self.offsprings = []

    def local_search(self):
        """
        Apply local search to optimize the population.
        """
        for ind in self.elites:
            new_fitness = local_search_swap(ind.perm)
            ind.fitness = new_fitness

            if not ind.optimized:
                new_fitness = local_search_inversion(ind.perm, ind.fitness, ind.k)
                if not new_fitness is None:
                    ind.optimized = False
                    ind.fitness = new_fitness
                else: # There are no inversions of length k that can be done to decrease the fitness
                    ind.k += +1
                    if ind.k >= TSP.n:
                        ind.optimized = True

    def update_statistics(self):
        """
        Return a tuple containing the following:
            - the mean fitness of the population
            - the best fitness of the population
            - a 1D numpy array in the cycle notation containing the best solution
              with city numbering starting from 0
        :return: A tuple (m, bo, bs) that represent the mean objective, best objective and best solution respectively
        """
        mean = 0
        best_fitness = -1
        worst_fitness = -1
        best_individual = None
        for ind in self.elites + self.peasants:
            f = ind.fitness
            mean += f
            if f < best_fitness or best_fitness == -1:
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

class Individual:
    """
    A class that represents an individual in the population.
    This class will contain an order in which to visit the cities.
    """

    def __init__(self, perm, f, alpha, beta, k=2, optimized=False):
        """
        Create a new TSPTour with the specified parameters
        :param perm: The permutation, this should be a numpy.array containing the order of the cities.
        :param f: The fitness value for this individual.
        :param alpha: The probability of mutation.
        :param beta: The probability of crossover.
        :param k: The length of the inversions to search through in the local search.
        """
        self.perm = perm
        self.fitness = f
        self.next_cities = np.zeros(TSP.n, dtype=int)
        for i in range(TSP.n):
            self.next_cities[self.perm[i]] = self.perm[(i+1) % TSP.n]
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.optimized = optimized

    def cpy(self):
        return Individual(np.copy(self.perm), self.fitness, self.alpha, self.beta, self.k, self.optimized)


### ----------- ###
###  OPERATORS  ###
### ----------- ###

def select(population,k):
    """
    Select an individual from the population using K-tournament selection.
    :param population: The population to choose from.
    :param k: The parameter for the K-tournament selection.
    :return: An individual from the population selected with K-tournament selection.
    """
    return min(rnd.choices(population, k=k), key=lambda ind: ind.fitness)

def eliminate(population, la, q):
    """
    Eliminate a part of the given population to eventually become a population of the given size.
    :param population: The population to reduce.
    :param la: The size of the population after the elimination.
    :param q: The parameter for the crowding mechanism.
    :return: A list of size la that contains the survivors.
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    survivors = []
    while len(survivors) < la:
        survivor = sorted_pop.pop(0)
        survivors.append(survivor)
        if len(sorted_pop) > la - len(survivors):
            victim = min(rnd.sample(sorted_pop, k=min(q, len(sorted_pop))), key=lambda ind: distance(ind, survivor))
            sorted_pop.remove(victim)
        else:
            survivors += sorted_pop
            break

    return survivors

def local_search_swap(perm):
    """
    Search for a neighbouring permutation that has a better fitness by looking for a good swap.
    :param perm: The permutation to improve.
    :return: The fitness of the improved permutation or None if the permutation was not improved.
    """
    old_f = fitness(perm)
    #worst_edge = max(range(TSP.n), key=lambda i: TSP.distance_matrix[perm[i]][perm[(i+1) % TSP.n]])
    worst_edge = rnd.randrange(0, TSP.n)
    best_edge = min([i for i in range(TSP.n)], key= lambda i:fitness_swap(perm, old_f, worst_edge, i))
    perm[worst_edge], perm[best_edge] = perm[best_edge], perm[worst_edge]
    return fitness(perm)


def local_search_inversion(perm, old_fitness, max_length):
    """
    Search for a neighbouring permutation that has a better fitness by looking for a good inversion.
    :param perm: The permutation to improve.
    :param old_fitness: The fitness to improve.
    :param max_length: The maximum length of the inversions.
    :return: The fitness of the improved permutation or None if the permutation was not improved.
    """
    inversions = [(i, (i+max_length) % TSP.n) for i in range(TSP.n)]
    best_inversion = None
    best_fitness = old_fitness
    for i,j in inversions:
        new_fitness = fitness_inversion(perm, old_fitness, i, j)
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            best_inversion = (i,j)

    if not best_inversion is None:
        invert(perm, best_inversion[0], best_inversion[1])
        return fitness(perm)

    return None

def divide(population, epsilon, min_distance):
    """
    Divide the given population in elites and peasants.
    :param population: The population of Individuals to divide.
    :param epsilon: The size of the elite population.
    :param min_distance: The minimal distance the members of the elite group should have
    :return: a tuple (elites,peasants).
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    elites = []
    k = 0
    while len(elites) < epsilon and k < len(sorted_pop):
        allowed = True
        for elite in elites:
            if distance(sorted_pop[k], elite) < min_distance:
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
    city_used = np.zeros(shape=TSP.n, dtype=bool)
    start = rnd.randrange(0, TSP.n)

    perm_offspring[0] = start
    city_used[start] = True

    for i in range(1, TSP.n):
        c = perm_offspring[i - 1]
        c1 = next_cities1[c]
        c2 = next_cities2[c]

        c1_ok = not city_used[c1]
        c2_ok = not city_used[c2]
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
            available_cities = np.argwhere(city_used == False)
            perm_offspring[i] = available_cities[int(rnd.random() * len(available_cities))][0]

        city_used[perm_offspring[i]] = True
    return perm_offspring

def mutate(perm, max_length):
    """
    Perform Inversion Mutation on the given permutation.
    :param perm: The numpy.array to permute.
    :param max_length: The maximum length of the inversion.
    """
    start = rnd.randrange(TSP.n)
    end = (start + rnd.randrange(1, max_length + 1)) % TSP.n
    invert(perm, start, end)


### --------- ###
###  METRICS  ###
### --------- ###

def distance(ind1, ind2):
    """
    Calculate the distance between two individuals.
    :param ind1: The first individual.
    :param ind2: The second individual.
    :return: The distance between the two given individuals.
    """
    return TSP.n - 1 - common_edges(ind1.next_cities, ind2.next_cities)

def fitness(perm):
    """
    Calculate the fitness value of the given tour.
    :param perm: The order of the cities.
    :return: The fitness value of the given tour.
    """
    return np.sum(np.array([TSP.distance_matrix[perm[i % TSP.n]][perm[(i + 1) % TSP.n]] for i in range(TSP.n)]))

def fitness_inversion(perm, old_fitness, i, j):
    """
    Calculate the fitness of the given permutation if an inversion was performed.
    :param perm: The permutation to calculate to potential fitness for
    :param old_fitness: The current fitness of the permutation
    :param i: The first index of the inversion
    :param j: The second index of the inversion
    :return: The fitness of the given permutation if invert(perm,i,j) was called.
    """
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

def fitness_swap(perm, old_f, i, j):
    """
    Calculate the fitness if the given perm if the ith city was swapped with the jth city.
    :param perm: The permutation to calculate the potential fitness for.
    :param old_f: The current fitness of the permutation
    :param i: The index of the first city
    :param j: The index of the second city
    :return: The fitness if the given perm if the ith city was swapped with the jth city.
    """
    if i == j:
        return old_f

    i, j = min(i, j), max(i, j)
    if i == 0 and j == TSP.n-1:
        i,j = j,i

    old_f -= TSP.distance_matrix[perm[i]][perm[(i + 1) % TSP.n]]
    old_f -= TSP.distance_matrix[perm[(i - 1) % TSP.n]][perm[i]]
    old_f -= TSP.distance_matrix[perm[j]][perm[(j + 1) % TSP.n]]
    if j == (i+1) % TSP.n or (j+1) % TSP.n == i:
        old_f += TSP.distance_matrix[perm[j]][perm[i]]
    else:
        old_f -= TSP.distance_matrix[perm[(j - 1) % TSP.n]][perm[j]]
    old_f += TSP.distance_matrix[perm[(i - 1) % TSP.n]][perm[j]]
    old_f += TSP.distance_matrix[perm[j]][perm[(i + 1) % TSP.n]]
    old_f += TSP.distance_matrix[perm[(j - 1) % TSP.n]][perm[i]]
    old_f += TSP.distance_matrix[perm[i]][perm[(j + 1) % TSP.n]]
    return old_f


### ------------------- ###
###  PARAMETER CLASSES  ###
### ------------------- ###

class AlgorithmParameters:
    """
    Class that contains all the information to run the genetic algorithm.
    """

    def __init__(self, la, init_alpha, var_alpha, init_beta, island_params, exchange_size, exchange_rate, max_slope, slope_weight):
        """
        Create a new AlgorithmParameters object.
        :param la: The size of the entire population that will be divided over the islands
        :param init_alpha: The initial value of alpha, the probability of mutation
        :param var_alpha: The maximal random offset that will be applied to the initial alpha (in absolute value)
        :param init_beta: The initial value of beta, the probability of crossover
        :param island_params: The list of IslandParameter objects that specify the parameters for the islands
        :param exchange_size: A tuple (e,p) where e is the amount of elites to exchange and p the amount of peasants to exchange when an epoch finishes
        :param exchange_rate: The length of the epoch (specified in iterations)
        :param max_slope: The maximum value of the slope of the convergence graph
        :param slope_weight: The influence of new updates of the slope
        """
        self.la = la
        self.init_alpha = init_alpha
        self.var_alpha = var_alpha
        self.init_beta = init_beta
        self.island_params = island_params
        self.ex_size_elite = exchange_size[0]
        self.ex_size_peasants = exchange_size[1]
        self.exchange_rate = exchange_rate
        self.max_slope = max_slope
        self.slope_weight = slope_weight

class IslandParameters:
    """
    Class that contains all the information to run an island.
    """
    def __init__(self, mu, epsilon, q, min_dist, k, distribution, parent_reward):
        """
        Create a new IslandParameters object
        :param mu: The amount of offsprings to create
        :param epsilon: The size of the elite population
        :param q: The amount of individuals to choose from for the crowding mechanism
        :param min_dist: The minimum distance s to that distance(e1,e2) >= d for each e1,e2 in the elite population
        :param k: The parameter for k-tournament selection
        :param distribution: A list [p1,p2] where p1 is the proportion of offsprings that will be create from two elite parents,
                             and p2 the proportion of offsprings create a from a peasant and an elite.
        :param parent_reward: The reward that a parent receives for creating an offspring that is better then itself.
                              The beta of the parent will be multiplied with this value.
        """
        self.mu = mu
        self.epsilon = epsilon
        self.q = q
        self.min_dist = min_dist
        self.k = k
        self.distribution = distribution
        self.parent_reward = parent_reward


### ----------------- ###
###  UTILITY METHODS  ###
### ----------------- ###

def slc(x, i, j):
    """
    Return the x[i:j] but this also works for when j < i.
    :param x: The list or array to retrieve the slice from
    :param i: The first index (inclusive)
    :param j: the last index (exclusive)
    :return: x[i:j] if i < j, x[i:] + x[:j] if j < i
    """
    if j < i:
        return np.concatenate((x[i:], x[:j]))
    else:
        return x[i:j]

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

def common_edges(next_cities1, next_cities2):
    """
    Find the common edges in two tours
    :param next_cities1: A numpy array a where a[perm1[i]] = perm1[i+1 % n] with perm1 the corresponding tour
    :param next_cities2: A numpy array a where a[perm2[i]] = perm2[i+1 % n] with perm2 the corresponding tour
    :return:
    """
    return np.count_nonzero(next_cities1 == next_cities2)

def invert(perm, start, end):
    """
    Invert the given permutation.
    The result will be as follows:
        ... perm[start-1] perm[end-1] perm[end-2] ... perm[start+1] perm[start] perm[end] perm[end+1] ...
    :param perm: The permutation to invert.
    :param start: The first index (inclusive).
    :param end: The last index (exclusive).
    """
    if end > start:
        perm[start:end] = np.flip(perm[start:end])
    else:
        flipped_perm = np.concatenate((np.flip(perm[:end]), np.flip(perm[start:])))
        perm[start:] = flipped_perm[:(perm.shape[0] - start)]
        if end > 0:
            perm[:end] = flipped_perm[-end:]
