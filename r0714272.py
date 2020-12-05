import Reporter
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

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
		self.ap = AlgorithmParameters(la=50, mu=150, beta=0.9)
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
		self.n = distance_matrix.shape[0]				# The length of the tour
		self.population = []							# The list of Individual objects
		self.offsprings = []							# The list that will contain the offsprings

		self.slopes = []
		self.real_slopes = []
		self.so = SelectionOperator(max_k=max(10, int(0.1 * self.params.la) + 1), min_k=2)
		self.mo = MutationOperator(max_alpha=0.2, min_alpha=0.01, base_alpha=0.1, max_length=self.n, min_length=1, base_length=int(self.n / 10) + 1)
		self.ro = RecombinationOperator(distance_matrix)
		self.lso = LocalSearchOperator(objf=self.fitness, k=2, min_nbh=5, max_nbh=300, distance_matrix=distance_matrix)
		self.eo = EliminationOperator(keep=params.la, k=10, elite=1)
		#self.eo = EliminationOperator2(keep=params.la, q=10)
		self.cc = ConvergenceChecker(max_slope=-0.000001, weight=0.5)

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
			if self.mo.mutate(ind):
				ind.fitness = self.fitness(ind.perm)

	def elimination(self):
		"""
		Eliminate certain Individuals from the population.
		"""
		self.population = self.eo.eliminate(self.population, self.offsprings)
		self.offsprings = []

	def local_search(self):
		"""
		Apply local search to optimize the population.
		"""
		for ind in self.offsprings:
			self.lso.improve(ind)


	def has_converged(self):
		"""
		Check whether the algorithm has converged and should be stopped
		:return: True if the algorithm should stop, False otherwise
		"""
		return not self.cc.should_continue()

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

		objs = [ind.fitness for ind in self.population]
		best_obj = min(objs)
		worst_obj = max(objs)
		self.cc.update(best_obj)
		slope_progress = self.cc.get_slope_progress()
		self.so.update(slope_progress)
		self.mo.update(best_obj, worst_obj, slope_progress)
		self.lso.update(best_obj, worst_obj, slope_progress)
		#self.params.la = int((1 - slope_progress) * 100 + 30)
		#self.params.mu = int((1 - slope_progress) * 200 + 60)
		print("slope: {}".format(self.cc.slope))
		print("harshness: {}".format(self.mo.harshness))
		print("k: {}".format(self.so.k))
		print("nbh: {}".format(self.lso.nbh))
		#print("la: {}".format(self.params.la))
		#print("mu: {}".format(self.params.mu))
		self.local_search()
		self.create_offsprings()
		self.local_search()
		self.mutate()
		self.local_search()
		self.elimination()

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
		self.next_cities = np.zeros(self.n, dtype=int)
		for i in range(self.n):
			self.next_cities[self.perm[i]] = self.perm[(i+1) % self.n]

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
		self.alpha = max_k

	def update(self, slope_progress):
		"""
		Update the value of k.
		:param slope_progress: The progress of the slope
		"""
		alpha = 1 - slope_progress
		self.k = int(round(self.min_k + alpha * (self.max_k - self.min_k)))

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

	def __init__(self, max_alpha, min_alpha, base_alpha, max_length, min_length, base_length):
		"""
		Create a new SelectionOperator.
		:param max_alpha: The maximal value for parameter alpha.
		:param min_alpha: The minimal value for parameter alpha.
		:param base_alpha: The base value for alpha.
		:param max_length: The maximal length of the tour to invert.
		:param min_length: The minimal length of the tour to invert.
		:param base_length: The base value for the length of the inversion.
		"""
		self.max_alpha = max_alpha
		self.min_alpha = min_alpha
		self.base_alpha = base_alpha
		self.max_length = max_length
		self.min_length = min_length
		self.base_length = base_length
		self.best_obj = 0
		self.worst_obj = 0
		self.harshness = 1	# This value indicates how much impact the mutation has on the individual

	def update(self, best_obj, worst_obj, slope_progress):
		"""
		Update the value of alpha and the length of the tour to invert.
		:param best_obj: The current best objective
		:param worst_obj: The current worst objective
		:param slope_progress: The current progress of the slope
		"""
		self.best_obj = best_obj
		self.worst_obj = worst_obj
		#self.harshness = 1 - slope_progress
		#if slope_progress == 0:
		#	self.harshness = 1
		#else:
		#	self.harshness = min(1, -2 * np.log10(slope_progress))
		#self.harshness = max(0, 1 / (60 * (slope_progress + (-60.5 / 60))) + 1 - (1 / 60.5))
		#if slope_progress <= 0.7:
		#self.harshness = - 2 * np.sqrt(-(slope_progress - 0.5) ** 2 + 0.25) + 1
		#else:
		#	self.harshness = 0.5 * (np.power(slope_progress, 1 / 0.026)) - 0.5 * (np.power(1, 1 / 0.026)) + 0.5

	def mutate(self, ind):
		"""
		Mutate the given Individual.
		:param ind: The Individual to mutate.
		:return: True if the given Individual was mutated, False otherwise.
		"""
		s = 1
		if self.worst_obj != self.best_obj:
			s = abs(ind.fitness - self.best_obj) / abs(self.worst_obj - self.best_obj)

		s = max(s ** (1 - self.harshness), self.harshness)

		alpha = max(self.min_alpha, min(self.max_alpha, self.min_alpha + s * (self.max_alpha - self.min_alpha)))
		if rnd.random() <= alpha:
			(start, end) = random_ind(ind.perm.shape[0])
			ind.perm[start:end] = np.flip(ind.perm[start:end])
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
		available_cities = list(range(self.n))
		available_cities.remove(perm_offspring[0])
		rnd.shuffle(available_cities)
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
				perm_offspring[i] = available_cities[0]

			available_cities.remove(perm_offspring[i])

		return perm_offspring

class LocalSearchOperator:
	""" Class that represents a local search operator. """

	def __init__(self, objf, k, min_nbh, max_nbh, distance_matrix):
		"""
		Create new LocalSearchOperator.
		:param objf: The objective function to use
		:param k: The value for the parameter used in k-opt local search.
		:param min_nbh: The minimal amount of neighbours to calculate.
		:param min_nbh: The maximal amount of neighbours to calculate.
		:param distance_matrix: The distance matrix
		"""
		self.objf = objf
		self.min_nbh = min_nbh
		self.max_nbh = max_nbh
		self.thoroughness = 0
		self.k = k
		self.nbh = min_nbh
		self.distance_matrix = distance_matrix
		self.n = distance_matrix.shape[0]
		self.best_obj = 0
		self.worst_obj = 0

	def update(self, best_obj, worst_obj, slope_progress):
		"""
		Update this LocalSearchOperator
		:param best_obj: The current best objective
		:param worst_obj: The current worst objective
		:param slope_progress: The current progress of the slope
		"""
		self.thoroughness = slope_progress
		self.nbh = int(round(self.min_nbh + self.thoroughness * (self.max_nbh - self.min_nbh)))
		self.best_obj = best_obj
		self.worst_obj = worst_obj

	def improve(self, ind):
		"""
		Improve the given Individual.
		:param ind: The Individual to improve.
		"""
		s = 1
		if self.worst_obj != self.best_obj:
			s = abs(ind.fitness - self.best_obj) / abs(self.worst_obj - self.best_obj)
		nbh = int(round(self.min_nbh + s * (self.max_nbh - self.min_nbh)))
		(best_swap, best_fitness) = self._get_best_neighbour(ind, nbh)
		if not best_swap is None:
			ind.perm[(best_swap[0]+1):(best_swap[1])] = np.flip(ind.perm[(best_swap[0]+1):(best_swap[1])])

	def _improve(self, ind, k):
		if k == 0:
			return

		nbs_t = self._get_neighbours(ind)
		nbs = [Individual(flip_copy(ind.perm, nb_t[0][0], nb_t[0][1]), nb_t[1]) for nb_t in nbs_t]
		for i in nbs:
			self._improve(i, k-1)

		best_nb = min(nbs, key= lambda indi: indi.fitness)
		if best_nb.fitness < ind.fitness:
			ind.fitness = best_nb.fitness
			ind.perm = best_nb.perm


	def _calc_fitness_swap(self, i, j, perm):
		"""
		Calculate the new fitness of given permutation if edge i and j were swapped.
		:param i: The first edge
		:param j: The second edge
		:param perm: The permutation to calculate the new fitness of
		:return: The new fitness of given permutation if edge i and j were swapped.
		"""
		b_dist1 = self.distance_matrix[perm[i]][perm[(i + 1) % self.n]]
		b_dist2 = self.distance_matrix[perm[j]][perm[(j + 1) % self.n]]
		n_dist1 = self.distance_matrix[perm[i]][perm[j]]
		n_dist2 = self.distance_matrix[perm[(i + 1) % self.n]][perm[(j + 1) % self.n]]

		return b_dist1+b_dist2, n_dist1 + n_dist2

	def _get_neighbours(self, ind):
		"""
		Get the neighbours of the given Individual.
		:param ind: The Individual to calculate the neighbours for.
		:return: A list of ((i,j), f) where (i,j) is the swap and f the fitness.
		"""
		swaps = [random_ind(ind.n) for _ in range(self.nbh)]
		nbs = []
		for i, j in swaps:
			(b_dist,n_dist) = self._calc_fitness_swap(i,j,ind.perm)
			fitness = ind.fitness - b_dist + n_dist
			nbs.append(((i,j),fitness))

		return nbs

	def _get_best_neighbour(self, ind, nbh):
		swaps = [random_ind(ind.n) for _ in range(nbh)]
		best_fitness = float('inf')
		best_swap = None
		for i, j in swaps:
			(b_dist,n_dist) = self._calc_fitness_swap(i,j,ind.perm)
			if b_dist < n_dist:
				fitness = ind.fitness - b_dist + n_dist
				if fitness < best_fitness:
					best_fitness = fitness
					best_swap = (i,j)


		return best_swap, best_fitness

class ConvergenceChecker:
	""" A class for checking if the population has converged. """

	def __init__(self, max_slope, weight):
		"""
		Create a new ConvergenceChecker.
		:param max_slope: The minimal slope that the convergence graph should have.
		"""
		self.max_slope = max_slope
		self.best_objs = []
		self.slope = float("-inf")
		self.weight = weight

	def update(self, best_obj):
		"""
		Update this ConvergenceChecker
		:param best_obj: The current best objective
		"""
		self.best_objs.append(best_obj)
		if len(self.best_objs) == 2:
			self.slope = self.best_objs[1] - self.best_objs[0]
		elif len(self.best_objs) > 2:
			self.slope = (1 - self.weight) * self.slope + self.weight * (self.best_objs[-1] - self.best_objs[-2])

	def get_slope_progress(self):
		"""
		Get a value between 0 and 1 indicating how far the slope is from its target.
		:return: A value between 0 and 1 indicating how far the slope is from its target.
		"""
		if len(self.best_objs) < 2:
			return 0
		min_slope = self.best_objs[0] - self.best_objs[1]
		return 1 - abs(self.max_slope - self.slope) / abs(self.max_slope - min_slope)

	def should_continue(self):
		"""
		Check if the algorithm shoud continue.
		:return: True if the algorithm should continue, False otherwise.
		"""
		return self.slope <= self.max_slope

class EliminationOperator:
	""" Class that represents an elimination operator. """

	def __init__(self, keep, k, elite):
		"""
		Create a new EliminationOperator.
		:param keep:
		:param k: The amount of individuals to sample for choosing the victim.
		:param elite: The amount of individuals that go on to the next generation without doubt
		"""
		self.keep = keep
		self.k = k
		self.elite = elite

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

	def eliminate(self, parents, offsprings):
		"""
		Eliminate certain Individuals from the given population.
		:param parents: The parents of the population.
		:param offsprings: The offsprings of the population
		:return: The reduced population.
		"""
		new_population = []
		sorted_parents = sorted(parents, key= lambda ind: ind.fitness)
		sorted_offsprings = sorted(offsprings, key= lambda ind: ind.fitness)

		elites = [sorted_parents[i] for i in range(self.elite) if sorted_parents[i].fitness < sorted_offsprings[0].fitness]
		new_population += elites

		for _ in range(len(elites), self.keep):
			if len(sorted_offsprings) > 0:
				survivor = sorted_offsprings.pop(0)
				new_population.append(survivor)
				victims = rnd.choices(sorted_offsprings, k=min(self.k, len(sorted_offsprings)))
				if len(victims) > 0:
					sorted_offsprings.remove(min(victims, key= lambda ind: EliminationOperator.distance(ind.perm, survivor.perm)))
			else:
				new_population += sorted_parents[len(elites):len(elites) + self.keep - len(new_population)]
		return new_population

class EliminationOperator2:
	""" Class that represents another elimination operator. """

	def __init__(self, keep, q):
		"""
		Create a new EliminationOperator.
		:param keep: The amount of individuals to keep.
		:param q: the amount of battles to organise
		"""
		self.keep = keep
		self.q = q

	def eliminate(self, parents, offsprings):
		"""
		Eliminate certain Individuals from the given population.
		:param parents: The parents of the population.
		:param offsprings: The offsprings of the population.
		:return: The reduced population.
		"""
		pop = parents + offsprings
		wins = np.zeros(len(pop))
		for i,ind in enumerate(pop):
			opponents = rnd.choices(pop, k=self.q)
			for opp in opponents:
				if opp.fitness < ind.fitness:
					wins[i] += 1
		survivors = sorted(enumerate(pop), key= lambda t: wins[t[0]])[0:self.keep]
		return [ind for _,ind in survivors]

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