from math import isinf

import Reporter
import numpy as np
import random as rnd
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
		print("SEED = {}".format(seed))
		rnd.seed(seed)
		np.random.seed(seed)

		island_params = [IslandParameters(mu=200, epsilon=2, q=5, min_dist=5, k=3)] * 4
		params = AlgorithmParameters(la=200, init_alpha=0.1, init_beta=0.9, island_params=island_params, exchange_size=10, exchange_rate=30)
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
	def __init__(self, distance_matrix, params):
		TSP.distance_matrix = distance_matrix
		self.n = distance_matrix.shape[0]
		self.params = params
		self.islands = []
		self.sc = SlopeChecker(-0.0000000001, 0.5)

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
			population.sort(key=lambda ind: distance(ind.perm, leader.perm))
			island_pop = [population.pop(0) for _ in range(island_la - 1)]
			island_pop.append(leader)
			self.islands.append(Island(island_pop, island_params))

	def get_initial_permutations(self):
		permutations = []
		for _ in range(self.params.la):
			start = rnd.randrange(0, self.n)
			available_cities = set(range(self.n))
			perm = np.zeros(self.n, dtype=int)
			perm[0] = start
			available_cities.remove(start)
			for i in range(1, self.n):
				perm[i] = min(available_cities, key=lambda city: self.distance_matrix[perm[i - 1]][city])
				available_cities.remove(perm[i])
			(i, j) = random_ind(self.n)
			flip(perm, i, j)
			permutations.append(perm)
		return permutations

	def distribute_population(self):
		for i in range(len(self.islands)):
			victims = [self.islands[i].population.pop(rnd.randrange(0, len(self.islands[i].population))) for _ in range(self.params.exchange_size)]
			self.islands[(i+1) % len(self.islands)].population += victims

	def update(self):
		print("-------------------------------------")
		best_obj = min([island.best_individual.fitness for island in self.islands])
		self.sc.update(best_obj)

		for island in self.islands:
			#print("best: {} \t mean: {} \t worst: {}".format(island.best_individual.fitness, island.mean_fitness, island.worst_fitness))
			island.update()

		best_ind = min([island.best_individual for island in self.islands], key=lambda ind: ind.fitness)
		print("best: {}, alpha: {}, beta: {}".format(best_ind.fitness, best_ind.alpha, best_ind.beta))

		#if self.iteration % 30 == 0: #.should_redistribute:
		#	print("Redistributing population...")
		#	self.distribute_population()
		#	self.should_redistribute = False

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
		self.distribution = [0.1, 0.2]
		self.update_statistics()

	def create_offsprings(self):

		#elite_pool = [(elite1, elite2) for elite1 in self.elites for elite2 in self.elites]
		elite_pool = [(select(self.elites, self.params.k), select(self.elites, self.params.k)) for _ in range(int(self.distribution[0] * self.params.mu) + 1)]
		self.do_crossovers(elite_pool)

		elite_peasants_pool = [(select(self.elites, self.params.k), select(self.peasants, self.params.k)) for _ in range(int(self.distribution[1] * self.params.mu) + 1)]
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
		for ind in self.offsprings:
			if rnd.random() < ind.alpha:
				mutate(ind.perm, ind.perm.shape[0]-1) #TODO: what should max_length be?
				ind.fitness = fitness(ind.perm)

	def elimination(self):
		"""
		Eliminate certain Individuals from the population.
		"""
		o_elites, o_peasants = divide(self.offsprings, self.params.epsilon, self.params.min_dist)
		new_population = eliminate(self.elites, self.peasants, o_elites, o_peasants, self.params.q, self.la)
		self.elites, self.peasants = divide(new_population, self.params.epsilon, self.params.min_dist)
		self.offsprings = []

	def local_search(self):
		"""
		Apply local search to optimize the population.
		"""
		for ind in self.elites:
			new_perm, new_fitness = local_search(ind.perm, ind.fitness, ind.k)
			if not new_perm is None:
				ind.perm = new_perm
				ind.fitness = new_fitness
				#ind.k = 1
			else:
				ind.k = min(ind.k + 1, ind.perm.shape[0] - 1)


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
		print("elites: {}, peasants: {}".format(len(self.elites), len(self.peasants)))

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
		self.n = self.perm.shape[0]
		self.next_cities = np.zeros(self.n, dtype=int)
		for i in range(self.n):
			self.next_cities[self.perm[i]] = self.perm[(i+1) % self.n]
		self.alpha = alpha
		self.beta = beta
		self.k = 2

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

def eliminate(elites, peasants, o_elites, o_peasants, q, la):
	sorted_pop = sorted(elites + peasants + o_elites + o_peasants, key=lambda ind: ind.fitness)
	survivors = []

	while len(survivors) < la:
		survivor = sorted_pop.pop(0)
		survivors.append(survivor)
		victim = min(rnd.choices(sorted_pop, k=min(q, len(sorted_pop))), key=lambda ind: ind.fitness)
		sorted_pop.remove(victim)

	return survivors


	#wins_elites = battle(elites, o_elites, q)
	#wins_peasants = battle(peasants, o_peasants, q)

	#elites = sorted(wins_elites.keys(), key=lambda ind: wins_elites[ind], reverse=True)[:len(elites)]
	#peasants = sorted(wins_peasants.keys(), key=lambda ind: wins_peasants[ind], reverse=True)[:len(peasants)]

	#return elites, peasants

def battle(team1, team2, q):
	wins = dict()
	for tm1 in team1:
		opponents = rnd.choices(team2, k=q)
		wins[tm1] = np.sum(np.array([1 if tm1.fitness < o.fitness else 0 for o in opponents]))

	for tm2 in team2:
		opponents = rnd.choices(team1, k=q)
		wins[tm2] = np.sum(np.array([1 if tm2.fitness < o.fitness else 0 for o in opponents]))

	return wins

def local_search(perm, old_fitness, k):
	"""
	Search for a neighbouring permutation that has a better fitness.
	:param perm: The permutation to improve.
	:param old_fitness: The fitness to improve.
	:param k: The maximum length of the inversions.
	:return: A tuple (p,f) where p is new permutation that has better fitness than the given one or None if no such permutation can be found and f its fitness.
	"""
	n = perm.shape[0]
	inversions = [(i, (i+k) % n) for i in range(n)]
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
	n = TSP.distance_matrix.shape[0]
	return np.sum(np.array([TSP.distance_matrix[perm[i % n]][perm[(i + 1) % n]] for i in range(n)]))

def estimate_fitness(perm, old_fitness, i, j):
	n = perm.shape[0]
	new_fitness = old_fitness
	k = i
	while k % n != (j - 1) % n:
		new_fitness -= TSP.distance_matrix[perm[k % n]][perm[(k + 1) % n]]
		new_fitness += TSP.distance_matrix[perm[(k + 1) % n]][perm[k % n]]
		k += 1
	new_fitness -= TSP.distance_matrix[perm[(i - 1) % n]][perm[i]]
	new_fitness -= TSP.distance_matrix[perm[(j - 1) % n]][perm[j]]
	new_fitness += TSP.distance_matrix[perm[(i - 1) % n]][perm[(j - 1) % n]]
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
	n = TSP.distance_matrix.shape[0]
	perm_offspring = np.zeros(shape=n, dtype=int)
	perm_offspring_lu = set()
	start = rnd.randrange(0, n)
	perm_offspring[0] = start
	perm_offspring_lu.add(start)
	available_cities = set(range(n))
	available_cities.remove(perm_offspring[0])

	for i in range(1, n):
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
	start = rnd.randrange(perm.shape[0])
	end = (start + rnd.randrange(1, max_length + 1)) % perm.shape[0]
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
	return perm1.shape[0] - 1 - common_edges(perm1, perm2)

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
	n = a.shape[0]
	mp = {}
	for i in range(n):
		mp[b[i]] = i

	for i in range(n):
		b[i] = mp[a[i]]

	arrPos = [[0 for x in range(2)] for y in range(n)]

	for i in range(n):
		arrPos[i][0] = b[i]
		arrPos[i][1] = i

	arrPos.sort()
	vis = [False] * (n)

	ans = 0

	for i in range(n):

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
	n = a.shape[0]
	edges = set()
	for i in range(n):
		edges.add(frozenset([ a[i], a[(i+1) % n] ]))
		edges.add(frozenset([ b[i], b[(i+1) % n] ]))

	return 2 * n - len(edges)