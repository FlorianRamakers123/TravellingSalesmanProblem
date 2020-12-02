import Reporter
import numpy as np
import random as rnd

### TO-DO-LIST
# TODO: checkout round robin based elimination

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
		self.ap = AlgorithmParameters(la=500, mu=1000, beta=0.9)
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

	def __init__(self, distance_matrix, params, ):
		"""
		Create a new TSPAlgorithm object.
		:param distance_matrix: The distance matrix that contains the distances between all the cities.
		:param params: A AlgorithmParameters object that contains all the parameter values for executing the algorithm.
		"""
		self.distance_matrix = distance_matrix
		self.params = params
		self.n = distance_matrix.shape[0]				# The length of the tour
		self.population = []							# The list of Individual objects
		self.offsprings = []							# The list that will contain the offsprings

		self.so = SelectionOperator(max(3, int(0.1 * self.n) + 1), 2)
		self.mo = MutationOperator(0.2, 0.05, self.n, 2)
		self.ro = RecombinationOperator(distance_matrix)
		self.lso = LocalSearchOperator(self.fitness, 2, 5)
		self.eo = EliminationOperator(params.la, 2)
		self.cc = ConvergenceChecker(10)

	def fitness(self, perm):
		"""
		Calculate the fitness value of the given tour.
		:param perm: The order of the cities.
		:return: The fitness value of the given tour.
		"""
		return np.sum(np.array([self.distance_matrix[perm[i % self.n], perm[(i + 1) % self.n]] for i in range(self.n)]))

	def initialize(self):
		"""
		Initialize the population using random permutations and the initial values specified in the AlgorithmParameters object.
		"""
		permutations = [np.random.permutation(self.n) for _ in range(self.params.la)]
		self.population = [Individual(permutations[i], self.fitness(permutations[i])) for i in range(self.params.la)]

	def create_offsprings(self):
		""" Select 2 * mu parents from the population and apply a recombination operator on them. """
		while len(self.offsprings) < self.params.mu:
			if rnd.random() <= self.params.beta:
				p1,p2 = self.so.select(self.population), self.so.select(self.population)
				perm_offspring = self.ro.recombine(p1,p2)
				self.offsprings.append(Individual(perm_offspring, self.fitness(perm_offspring)))

	def mutate(self):
		"""
		Mutate the population.
		"""
		for ind in self.offsprings:
			if self.mo.mutate(ind.perm):
				ind.fitness = self.fitness(ind.perm)

	def elimination(self):
		"""
		Eliminate certain Individuals from the population.
		"""
		self.population = self.eo.eliminate(self.offsprings)
		self.offsprings = []

	def local_search(self):
		"""
		Apply local search to optimize the population.
		"""
		for ind in self.offsprings:
			nbh = self.lso.get_neighbourhood(ind) + [ind]
			nbh = sorted(nbh, key=lambda i: i.fitness)
			ind.perm = nbh[0].perm
			ind.fitness = nbh[0].fitness


	def has_converged(self):
		"""
		Check whether the algorithm has converged and should be stopped
		:return: True if the algorithm should stop, False otherwise
		"""
		return not self.cc.should_continue(self.population)

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


	def update(self):
		"""
		Update the population.
		"""
		print("Creating offsprings...")
		self.create_offsprings()
		print("Mutating...")
		self.mutate()
		print("Local search...")
		self.local_search()
		print("Elimination...")
		self.elimination()
		(mean_obj, best_obj, _) = self.report_values()
		print("Updating...")
		self.so.update(best_obj, mean_obj)
		self.mo.update(best_obj, mean_obj)


class Individual:
	"""
	A class that represents an order in which to visit the cities.
	This class will represent an individual in the population.
	"""

	def __init__(self, perm, fitness):
		"""
		Create a new TSPTour with the specified parameters
		:param perm: The permutation, this should be a numpy.array containing the order of the cities.
		:param fitness: The fitness value for this individual.
		"""
		self.perm = perm
		self.fitness = fitness
		self.n = self.perm.shape[0]
		self.next_cities = { self.perm[i] : self.perm[(i+1) % self.n] for i in range(self.n) }

	def get_next(self, city):
		"""
		Get the city that follows next to the given city
		:param city: The number of the city (starting from zero)
		:return: The number of the city that follows the given city in this Individual.
		"""
		return self.next_cities[city]

class SelectionOperator:
	"""
	Class that represents a selection operator for a genetic algorithm.
	This operator is updated to make sure that we have large values for k in the beginning and small values near the end
	"""

	def __init__(self, max_k, min_k):
		"""
		Create a new SelectionOperator
		:param max_k: The maximal value for parameter for the k-tournament selection.
		:param min_k: The minimal value for parameter for the k-tournament selection.
		"""
		self.max_k = max_k
		self.min_k = min_k
		self.k = max_k

	def update(self, best_obj, mean_obj):
		"""
		Update the value of k.
		:param best_obj: The current best objective
		:param mean_obj: The current mean objective
		"""
		self.k = int(min(self.max_k, max(self.min_k, mean_obj / best_obj * self.min_k)))

	def select(self, population):
		"""
		Select a parent from the population for recombination.
		:param population: The population to choose from
		:return: An Individual object that represents a parent.
		"""
		selected = rnd.choices(population, k=self.k)
		selected = sorted(selected, key=lambda ind: ind.fitness)
		return selected[0]

class MutationOperator:
	"""
	Class that represents a mutation operator for a genetic algorithm.
	This operator is updated to make sure that we have large values for alpha in the beginning and small values near the end.
	"""

	def __init__(self, max_alpha, min_alpha, max_length, min_length):
		"""
		Create a new SelectionOperator
		:param max_alpha: The maximal value for parameter alpha.
		:param min_alpha: The minimal value for parameter alpha.
		:param max_length: The maximal length of the tour to invert
		:param min_length: The minimal length of the tour to invert
		"""
		self.max_alpha = max_alpha
		self.min_alpha = min_alpha
		self.alpha = max_alpha

		self.max_length = max_length
		self.min_length = min_length
		self.length = max_length

	def update(self, best_obj, mean_obj):
		"""
		Update the value of alpha and the length of the tour to invert.
		:param best_obj: The current best objective
		:param mean_obj: The current mean objective
		"""
		self.alpha = min(self.max_alpha, max(self.min_alpha, mean_obj / best_obj * self.min_alpha))
		self.length = round(min(self.max_length, max(self.min_length, mean_obj / best_obj * self.min_length)))

	def mutate(self, perm):
		"""
		Mutate the given permutation.
		:param perm: The permutation to mutate.
		:return: True if the given perm was mutated, False otherwise.
		"""
		if rnd.random() <= self.alpha: #TODO: alpha is no longer self-adapted, is this a good idea?
			(start, end) = random_ind(perm.shape[0])
			end -= max((end - start) - self.length, 0)
			perm[start:end] = np.flip(perm[start:end])
			return True
		return False

class RecombinationOperator:
	"""
	Class that represents a recombination operator for a genetic algorithm.
	"""

	def __init__(self, distance_matrix):
		self.distance_matrix = distance_matrix
		self.n = distance_matrix.shape[0]

	def recombine(self, parent1, parent2):
		"""
		:param parent1: The first parent permutation.
		:param parent2: The second parent permutation.
		:return: An permutation that represents the offspring of parent1 and parent2
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

		return perm_offspring

class LocalSearchOperator:
	""" Class that represents a local search operator. """

	def __init__(self, objf, k, nbh_limit):
		"""
		Create new LocalSearchOperator.
		:param objf: The objective function to use
		:param k: The parameter used in k-opt local search.
		:param nbh_limit: The maximal amount of neighbours to calculate.
		"""
		self.objf = objf
		self.k = k
		self.nbh_limit = nbh_limit

	def get_neighbourhood(self, ind):
		"""
		Get the entire neighbourhood of the given Individual.
		:param ind: The individual to calculate the neighbourhood for.
		:return: A list of individuals that represent the k-level neighbourhood of the given Individual.
		"""
		nbs = []
		idx = 0
		self._get_neighbours(ind, nbs)
		for i in range(self.k - 1):
			old_size = len(nbs) - idx
			for j in range(idx, len(nbs)):
				self._get_neighbours(nbs[j], nbs)
			idx += old_size
		return nbs


	def _get_neighbours(self, ind, nbs_list):
		"""
		Get the neighbours of the given Individual.
		:param ind: The Individual to calculate the neighbours for.
		:param nbs_list: The list to append the neighbours to.
		:return: A list of all individuals who are one swap away of this individual.
		"""
		swaps = [random_ind(ind.n) for _ in range(self.nbh_limit)]
		for (i, j) in swaps:
			perm = flip_copy(ind.perm, i, j)
			nbs_list.append(Individual(perm, self.objf(perm)))

class ConvergenceChecker:
	""" A class for checking if the population has converged. """

	def __init__(self, min_std):
		"""
		Create a new ConvergenceChecker.
		:param min_std: The minimal standard deviation that the population should have.
		"""
		self.min_std = min_std

	def should_continue(self, population):
		"""
		Check if the algorithm shoud continue.
		:param population: The population that is maintained by the algorithm.
		:return: True if the algorithm should continue, False otherwise.
		"""
		fitnesses = [ind.fitness for ind in population]
		return np.sqrt(np.var(fitnesses)) > self.min_std

class EliminationOperator:
	""" Class that represents an elimination operator. """

	def __init__(self, keep, k):
		"""
		Create a new EliminationOperator.
		:param keep:
		:param k: The amount of individuals to sample for choosing the victim.
		"""
		self.keep = keep
		self.k = k

	@staticmethod
	def distance(perm1, perm2):
		"""
		Calculate the distance between two permutations.
		:param perm1: The first permutations.
		:param perm2: The second permutations.
		:return: The distance between the two given permutations.
		"""
		start_idx = np.flatnonzero(perm2 == perm1[0])[0]
		if start_idx < perm1.shape[0] / 2:
			return min_swap(perm1, np.roll(perm2, -start_idx))
		else:
			return min_swap(perm1, np.roll(perm2, perm1.shape[0] - start_idx))

	def eliminate(self, population):
		"""
		Eliminate certain Individuals from the given population.
		:param population: The population to eliminate certain individuals from.
		:return: The reduced population.
		"""
		new_population = []
		sorted_pop = sorted(population, key= lambda ind: ind.fitness)

		for _ in range(self.keep):
			survivor = sorted_pop.pop(0)
			new_population.append(survivor)
			victims = rnd.choices(sorted_pop, k=self.k)
			sorted_pop.remove(min(victims, key= lambda ind: EliminationOperator.distance(ind.perm, survivor.perm)))

		return new_population

class AlgorithmParameters:
	"""
	A class that contains all the information to run the genetic algorithm.
	"""

	def __init__(self, la, mu, beta):
		"""
		Create a new AlgorithmParameters object.
		:param la: The population size
		:param mu: The amount of tries to create offsprings (not every try will result in offsprings)
		:param beta: The probability that two parents
		"""
		self.la = la
		self.mu = mu
		self.beta = beta


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