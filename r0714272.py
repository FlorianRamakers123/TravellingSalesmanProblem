import Reporter
import numpy as np
import random as rnd

### TO-DO-LIST
# TODO: the chance of mutation should drop towards the end
# TODO: checkout round robin based elimination
# TODO: the k should be large in the beginning and small near the end
# TODO: mutation should have large intervals in the beginning and small in the end

class r0714272:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.ap = None

	# The evolutionary algorithm's main loop
	def optimize(self, filename):

		# Read distance matrix from file.
		file = open(filename)
		distance_matrix = np.loadtxt(file, delimiter=",")
		file.close()
		self.ap = AlgorithmParameters(la=50, mu=100, init_alpha=0.1, init_beta=0.9, k=5, max_iter=500, min_std=0.01, std_tol=100, k_opt=2, gamma=0.0, min_dist=30, p_exp=2, nbh_limit=5)
		tsp = TSP(distance_matrix, self.ap)

		# Initialize the population
		tsp.initialize()

		while not tsp.has_converged():
			(mean_obj, best_obj, best_sol) = tsp.report_values()
			print("mean: {}, best: {}".format(mean_obj, best_obj))
			time_left = self.reporter.report(mean_obj, best_obj, best_sol)
			if time_left < 0:
				print("Ran out of time!")
				break

			tsp.update()

		return 0

class TSP:
	""" A class that represents the evolutionary algorithm used to find solutions to the Travelling Salesman Problem (TSP) """

	def __init__(self, distance_matrix, params):
		"""
		Create a new TSPAlgorithm object.
		:param distance_matrix: The distance matrix that contains the distances between all the cities.
		:param params: A AlgorithmParameters object that contains all the parameter values for executing the algorithm.
		"""
		self.distance_matrix = distance_matrix
		self.params = params
		self.n = distance_matrix.shape[0]
		self.population = []
		self.offsprings = []
		self.iterations = 0

		self.counter = 0 # used for std_tol
		self.distances = None

	def fitness(self, perm):
		"""
		Calculate the fitness value of the given tour.
		:param perm: The order of the cities.
		:return: The fitness value of the given tour.
		"""
		return np.sum(np.array([self.distance_matrix[perm[i % self.n], perm[(i + 1) % self.n]] for i in range(self.n)]))

	def penalty(self, perm):
		"""
		Calculate the penalty for the given permutation.
		:param perm: The permutation to calculate the penalty for.
		:return: The penalty for the given permutation.
		"""
		return 1
		#distances = [(y, self.distance(perm, y.perm)) for y in self.offsprings]
		#N = [(y,dist) for (y,dist) in distances if dist < self.params.min_dist]
		#return sum(1 - pow(dist / self.params.min_dist, self.params.p_exp) for (_,dist) in N)

	def distance(self, perm1, perm2):
		"""
		Calculate the distance between two permutations.
		:param perm1: The first permutations.
		:param perm2: The second permutations.
		:return: The distance between the two given permutations.
		"""
		start_idx = np.flatnonzero(perm2 == perm1[0])[0]
		if start_idx < self.n / 2:
			return min_swap(perm1, np.roll(perm2, -start_idx))
		else:
			return min_swap(perm1, np.roll(perm2, self.n - start_idx))

	def initialize(self):
		"""
		Initialize the population using random permutations and the initial values specified in the AlgorithmParameters object.
		"""
		permutations = [np.random.permutation(self.n) for _ in range(self.params.la)]
		self.population = [Individual(permutations[i], self.params.init_alpha, self.params.init_beta, self.fitness(permutations[i]), self.penalty(permutations[i])) for i in range(self.params.la)]

	def select(self):
		"""
		Select a parent from the population for recombination.
		### CURRENT IMPLEMENATION K-tournament
		:return: An Individual object that represents a parent.
		"""
		selected = rnd.choices(self.population, k=self.params.k)
		selected = sorted(selected, key=lambda ind: ind.fitness)
		return selected[0]

	def create_offsprings(self):
		""" Select 2 * mu parents from the population and apply a recombination operator on them. """
		parents = [(self.select(), self.select()) for _ in range(int(self.params.mu))]
		for (p1,p2) in parents:
			if rnd.random() <= p1.beta:
				self.offsprings.append(self.recombine(p1,p2))
			#if rnd.random() <= p2.beta:
			#	self.offsprings.append(self.recombine(p2,p1))

	# TODO: If recombine works better delete this
	def recombine0(self, parent1, parent2):
		"""
		### CURRENT IMPLEMENTATION: order crossover
		:param parent1: The first parent.
		:param parent2: The second parent.
		:return: An Individual object that represents the offspring of parent1 and parent2
		"""
		length = parent1.perm.shape[0]
		(start, end) = random_ind(length)

		offspring_perm = np.ones(length).astype(int)
		offspring_perm[start:end + 1] = parent1.perm[start:end + 1]

		k = 0

		for i in range(length):
			if parent2.perm[(end + i + 1) % length] not in offspring_perm:
				offspring_perm[(end + k + 1) % length] = parent2.perm[(end + i + 1) % length]
				k += + 1

		c1 = 2 * (rnd.random() - 0.5)
		new_alpha = parent1.alpha + c1 * (parent2.alpha - parent1.alpha)
		c2 = 2 * (rnd.random() - 0.5)
		new_beta = parent1.beta + c2 * (parent2.beta - parent1.beta) #TODO: should beta be self-adapted?
		return Individual(offspring_perm, new_alpha, new_beta, self.fitness(offspring_perm), self.penalty(offspring_perm))

	def recombine(self, parent1, parent2):
		"""
		### CURRENT IMPLEMENTATION: heuristic crossover
		:param parent1: The first parent.
		:param parent2: The second parent.
		:return: An Individual object that represents the offspring of parent1 and parent2
		"""
		start = rnd.randrange(0, self.n)
		perm_offspring = np.zeros(shape=self.n, dtype=int)
		perm_offspring[0] = parent1.perm[start]
		for i in range(1,self.n):
			c = perm_offspring[i-1]
			c1 = parent1.get_next(c)
			c2 = parent2.get_next(c)

			c1_ok = c1 not in perm_offspring[0:i]
			c2_ok = c2 not in perm_offspring[0:i]
			if c1_ok and c2_ok:
				if self.distance_matrix[c][c1] < self.distance_matrix[c][c2]:
					perm_offspring[i] = c1
				else:
					perm_offspring[i] = c2
			elif c1_ok:
				perm_offspring[i] = c1
			elif c2_ok:
				perm_offspring[i] = c2
			else:
				p = rnd.randrange(0, self.n)
				while p in perm_offspring[0:i]:
					p = rnd.randrange(0, self.n)
				perm_offspring[i] = p

		c1 = 2 * (rnd.random() - 0.5)
		new_alpha = parent1.alpha + c1 * (parent2.alpha - parent1.alpha)
		c2 = 2 * (rnd.random() - 0.5)
		new_beta = parent1.beta + c2 * (parent2.beta - parent1.beta) #TODO: should beta be self-adapted?
		return Individual(perm_offspring, new_alpha, new_beta, self.fitness(perm_offspring), self.penalty(perm_offspring))

	def mutate(self):
		"""
		Mutate the population.
		"""
		for ind in self.offsprings:
			if rnd.random() <= ind.alpha:
				ind.mutate()
				ind.fitness = self.fitness(ind.perm)
				ind.penalty = self.penalty(ind.perm)

	def elimination(self):
		"""
		Eliminate certain Individuals from the population.
		### CURRENT IMPLEMENTATION: lambda, mu elimination + fitness sharing
		"""
		self.population = sorted(self.offsprings, key= lambda ind: ind.fitness * ind.penalty)[0:self.params.la]
		self.offsprings = []

	def local_search(self):
		"""
		Apply local search to optimize the population.
		### CURRENT IMPLEMENTATION: k-opt search
		"""
		for ind in self.offsprings:
			if rnd.random() < self.params.gamma:
				nbh = self.get_neighbourhood(ind) + [ind]
				nbh = sorted(nbh, key=lambda i: i.fitness)
				ind.perm = nbh[0].perm
				ind.alpha = nbh[0].alpha
				ind.beta = nbh[0].beta
				ind.fitness = nbh[0].fitness
				ind.penalty = self.penalty(nbh[0].perm)

	def has_converged(self):
		"""
		Check whether the algorithm has converged and should be stopped
		### CURRENT IMPLEMENTATION: iteration count
		:return: True if the algorithm should stop, False otherwise
		"""
		return self.iterations >= self.params.max_iter or self.counter > self.params.std_tol

	def get_neighbourhood(self, ind):
		"""
		Get the entire neighbourhood of the given Individual.
		:param ind: The individual to calculate the neighbourhood for.
		:return: A list of individuals that represent the k-level neighbourhood of the given Individual.
		"""
		nbs = []
		idx = 0
		self.get_neighbours(ind, nbs)
		for i in range(self.params.k_opt - 1):
			old_size = len(nbs) - idx
			for j in range(idx, len(nbs)):
				self.get_neighbours(nbs[j], nbs)
			idx += old_size
		return nbs


	def get_neighbours(self, ind, nbs_list):
		"""
		Get the neighbours of the given Individual.
		:param ind: The Individual to calculate the neighbours for.
		:param nbs_list: The list to append the neighbours to.
		:return: A list of all individuals who are one swap away of this individual.
		"""
		swaps = [random_ind(self.n) for _ in range(self.params.nbh_limit)]
		for (i, j) in swaps:
			perm = flip_copy(ind.perm, i, j)
			nbs_list.append(Individual(perm, ind.alpha, ind.beta, self.fitness(perm), -1))

	def report_values(self):
		"""
		Return a tuple containing the following:
			- the mean objective function value of the population
			- the best objective function value of the population
			- a 1D numpy array in the cycle notation containing the best solution
			  with city numbering starting from 0
		:return: A tuple (m, bo, bs) that represent the mean objective, best objective and best solution respectively
		"""
		mean = 0
		best_fitness = -1
		best_individual = None
		for ind in self.population:
			f = ind.fitness
			mean += f
			if f < best_fitness or best_fitness == -1:
				best_fitness = f
				best_individual = ind
		mean = mean / len(self.population)
		return mean, best_fitness, best_individual.perm

	"""
	Update the population.
	"""
	def update(self):
		print("Creating offsprings...")
		self.create_offsprings()
		print("Mutating...")
		self.mutate()
		print("Local search...")
		self.local_search()
		print("Elimination...")
		self.elimination()

		self.iterations += 1
		fitnesses = [ind.fitness for ind in self.population]
		if np.sqrt(np.var(fitnesses)) < self.params.min_std:
			self.counter += 1
		else:
			self.counter = 0

class Individual:
	"""
	A class that represents an order in which to visit the cities.
	This class will represent an individual in the population.
	"""

	def __init__(self, perm, alpha, beta, fitness, penalty):
		"""
		Create a new TSPTour with the specified parameters
		:param perm: The permutation, this should be a numpy.array containing the order of the cities.
		:param alpha: The probability that this individual will mutate.
		:param beta: The probability that this individual will create an offspring.
		:param fitness: The fitness value for this individual.
		:param beta: The penalty this Individual should receive.
		"""
		self.perm = perm
		self.alpha = alpha
		self.beta = beta
		self.fitness = fitness
		self.penalty = penalty
		self.n = self.perm.shape[0]
		self.next_cities = { self.perm[i] : self.perm[(i+1) % self.n] for i in range(self.n) }

	def mutate(self):
		"""
		### CURRENT IMPLEMENTATION: inversion mutation
		"""
		(start, end) = random_ind(self.n)
		self.perm[start:end] = np.flip(self.perm[start:end])

	def get_next(self, city):
		"""
		Get the city that follows next to the given city
		:param city: The number of the city (starting from zero)
		:return: The number of the city that follows the given city in this Individual.
		"""
		return self.next_cities[city]

class AlgorithmParameters:
	"""
	A class that contains all the information to run the genetic algorithm.
	Attributes:
		* la: The population size
		* mu: The amount of tries to create offsprings (not every try will result in offsprings)
		* init_alpha: The initial vlaue for alpha (the probability of mutation)
		* init_beta: The initial value for beta (the probability of recombination)
		* gamma: The probability of performing local search. # TODO: should this be self-adapted?
		### THESE MIGHT CHANGE
		* k: the parameter for the k-tournament selection
		* max_iter: the maximum amount of iterations the algorithm can run
		* min_std: the minimal standard deviation the population must have
		* std_tol: the maximum amount of iterations the standard deviations can be lower than min_std
		* k_opt: the parameter used in k-opt local search
		* min_dist: the minimal distance two Individuals must have in order to not receive a penalty.
		* p_exp: the exponent that is used in the fitness sharing penalty calculation
		* nbh_limit: the amount of neighbours to calculate
	"""

	def __init__(self, la, mu, init_alpha, init_beta, gamma, k, max_iter, min_std, std_tol, k_opt, min_dist, p_exp, nbh_limit):
		self.la = la
		self.mu = mu
		self.init_alpha = init_alpha
		self.init_beta = init_beta
		self.gamma = gamma
		self.k = k
		self.max_iter = max_iter
		self.min_std = min_std
		self.std_tol = std_tol
		self.k_opt = k_opt
		self.min_dist = min_dist
		self.p_exp = p_exp
		self.nbh_limit = nbh_limit


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