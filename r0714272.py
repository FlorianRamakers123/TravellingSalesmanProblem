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
		rnd.seed(1923971)
		np.random.seed(1923971)

		tsp = TSP(distance_matrix)

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

		its = list(range(len(tsp.island_bests[0])))
		for i in range(len(tsp.island_bests)):
			plt.plot(its, tsp.island_bests[i], label="Island {}".format(i+1))

		plt.legend()
		plt.show()

		return 0

class TSP:
	""" Class that represents an evolutionary algorithm to solve the Travelling Salesman Problem. """

	def __init__(self, distance_matrix):
		self.distance_matrix = distance_matrix
		self.n = distance_matrix.shape[0]
		self.islands = []
		self.island_bests = [[],[],[],[]]
		self.iteration = 0
		self.population = []
		self.cc = ConvergenceChecker(max_slope=-0.0000001, weight=0.8)
		self.la = 100
		self.mu = 200


		### ISLAND 1
		ap1 = AlgorithmParameters(self.la // 4, self.mu // 4, 1)
		so1 = KTournamentSelectionOperator(min_k=2, max_k=5)
		mo1 = InversionMutationOperator(min_alpha=0.15, max_alpha=0.2)
		ro1 = HeuristicCrossoverOperator(distance_matrix)
		lso1 = KOptLocalSearchOperator(objf=self.fitness, k=2, min_nbh=1, max_nbh=10, distance_matrix=distance_matrix)
		eo1 = EliminationOperator(keep=ap1.la, k=ap1.la, elite=5, df=TSP.distance)
		self.islands.append(Island(distance_matrix, ap1, so1, mo1, ro1, lso1, eo1, self.fitness, self))

		### ISLAND 2
		ap2 = AlgorithmParameters(self.la // 4, self.mu // 4, 0.9)
		so2 = KTournamentSelectionOperator(min_k=2, max_k=5)
		mo2 = InversionMutationOperator(min_alpha=0.1, max_alpha=0.2)
		ro2 = HeuristicCrossoverOperator(distance_matrix)
		lso2 = KOptLocalSearchOperator(objf=self.fitness, k=2, min_nbh=1, max_nbh=20, distance_matrix=distance_matrix)
		eo2 = EliminationOperator(keep=ap2.la, k=ap2.la, elite=5, df=TSP.distance)
		self.islands.append(Island(distance_matrix, ap2, so2, mo2, ro2, lso2, eo2, self.fitness, self))

		### ISLAND 3
		ap3 = AlgorithmParameters(self.la // 4, self.mu // 4, 0.8)
		so3 = KTournamentSelectionOperator(min_k=2, max_k=5)
		mo3 = InversionMutationOperator(min_alpha=0.1, max_alpha=0.2)
		ro3 = HeuristicCrossoverOperator(distance_matrix)
		lso3 = KOptLocalSearchOperator(objf=self.fitness, k=2, min_nbh=1, max_nbh=50, distance_matrix=distance_matrix)
		eo3 = EliminationOperator(keep=ap3.la, k=ap3.la, elite=5, df=TSP.distance)
		self.islands.append(Island(distance_matrix, ap3, so3, mo3, ro3, lso3, eo3, self.fitness, self))

		### ISLAND 4
		ap4 = AlgorithmParameters(self.la // 4, self.mu // 4, 0.8)
		so4 = KTournamentSelectionOperator(min_k=2, max_k=5)
		mo4 = InversionMutationOperator(min_alpha=0.01, max_alpha=0.2)
		ro4 = HeuristicCrossoverOperator(distance_matrix)
		lso4 = KOptLocalSearchOperator(objf=self.fitness, k=2, min_nbh=1, max_nbh=100, distance_matrix=distance_matrix)
		eo4 = EliminationOperator(keep=ap4.la, k=ap4.la, elite=5, df=TSP.distance)
		self.islands.append(Island(distance_matrix, ap4, so4, mo4, ro4, lso4, eo4, self.fitness, self))

	def initialize(self):
		"""
		Initialize the population using random permutations and the initial values specified in the AlgorithmParameters object.
		"""
		permutations = self.get_initial_permutations()
		for i in range(4):
			self.islands[i].population = [Individual(permutations[j + i*(self.la//4)], self.fitness(permutations[j+ i*(self.la//4)]), 0.1) for j in range(self.la // 4)]
		#self.population = [Individual(permutations[i], self.fitness(permutations[i])) for i in range(self.la)]
		self.distribute_population()

	def _get_initial_permutations(self):
		return [np.random.permutation(self.n) for _ in range(self.la)]

	def get_initial_permutations(self):
		permutations = []
		for _ in range(self.la):
			start = rnd.randrange(0, self.n)
			available_cities = set(range(self.n))
			perm = np.zeros(self.n, dtype=int)
			perm[0] = start
			available_cities.remove(start)
			for i in range(1, self.n):
				perm[i] = min(available_cities, key=lambda city: self.distance_matrix[perm[i-1]][city])
				available_cities.remove(perm[i])
			if len(np.unique(perm)) < len(perm):
				print(perm)
				raise RuntimeError("NOOOOOO")
			(i,j) = random_ind(self.n)
			perm[i+1:j+1] = np.flip(perm[i+1:j+1])
			permutations.append(perm)
		return permutations


	def distribute_population(self):
		#for i in range(4):
			#idx = rnd.randrange(0,len(pop))
			#leader = pop[idx]
			#pop.sort(key= lambda indi: TSP.distance(indi.perm, leader.perm))
			#group_members = pop[0:self.size // 4]
			#self.islands[i].population = group_members
			#for ind in group_members:
			#	pop.remove(ind)

		pop = []

		for i in range(4):
			pop += self.islands[i].population
		pop.sort(key=lambda ind: ind.fitness)

		for i in range(4):
			end = self.la - i * (self.la // 4)
			start = end - (self.la // 4)
			self.islands[i].population = pop[start:end]

		#for i in range(0, 3):
		#	self.islands[i].population.sort(key=lambda ind: ind.fitness)
		#	self.islands[i+1].population += self.islands[i].population

		#for i in range(3,0,-1):
		#	self.islands[i].population.sort(key=lambda ind: ind.fitness)
		#	self.islands[i-1].population += self.islands[i].population

	@staticmethod
	def distance(perm1, perm2):
		"""
		Calculate the distance between two permutations.
		:param perm1: The first permutations.
		:param perm2: The second permutations.
		:return: The distance between the two given permutations.
		"""
		#start_idx = np.flatnonzero(perm2 == perm1[0])[0]
		#if start_idx < perm1.shape[0] / 2:
		#	return min_swap(perm1, np.roll(perm2, -start_idx))
		#else:
		#	return min_swap(perm1, np.roll(perm2, perm1.shape[0] - start_idx))
		return perm1.shape[0] - 1 - common_edges(perm1, perm2)

	def update(self):
		self.iteration += 1

		print("--------------------- {} ---------------------".format(self.iteration))

		best_obj = min([island.report_values()[1] for island in self.islands])
		self.cc.update(best_obj)

		for island in self.islands:
			print("updating island...")
			island.update()


		if self.iteration % 10 == 0:
			print("Redistributing population...")
			self.distribute_population()
			values = [island.report_values() for island in self.islands]
			for i, (mo, bo, _) in enumerate(values):
				print("[{}]: best = {}, mean = {}".format(i+1, bo, mo))
				self.island_bests[i].append(bo)


	def report_values(self):
		values = [island.report_values() for island in self.islands]
		for i,(mo,bo,_) in enumerate(values):
			print("[{}]: best = {}, mean = {}".format(i+1,bo,mo))
			self.island_bests[i].append(bo)
		best = min(values, key=lambda v: v[1])
		return sum(v[0] for v in values) / 4.0, best[1], best[2]

	def has_converged(self):
		return not self.cc.should_continue()

	def fitness(self, perm):
		"""
		Calculate the fitness value of the given tour.
		:param perm: The order of the cities.
		:return: The fitness value of the given tour.
		"""
		return np.sum(np.array([self.distance_matrix[perm[i % self.n]][perm[(i + 1) % self.n]] for i in range(self.n)]))


class Island:
	""" A class that represents an island of an the evolutionary algorithm. """

	def __init__(self, distance_matrix, params, so, mo, ro, lso, eo, objf, tsp):
		"""
		Create a new Island object.
		:param distance_matrix: The distance matrix that contains the distances between all the cities.
		:param params: A AlgorithmParameters object that contains all the parameter values for executing the algorithm.
		"""
		self.distance_matrix = distance_matrix
		self.params = params
		self.n = distance_matrix.shape[0]				# The length of the tour
		self.population = None							# The list of Individual objects
		self.offsprings = []							# The list that will contain the offsprings
		self.objf = objf
		self.so = so
		self.mo = mo
		self.ro = ro
		self.lso = lso
		self.eo = eo
		self.tsp = tsp

	def create_offsprings(self):
		""" Select 2 * mu parents from the population and apply a recombination operator on them. """
		while len(self.offsprings) < self.params.mu:
			if rnd.random() <= self.params.beta:
				p1,p2 = self.so.select(self.population), self.so.select(self.population)
				perm_offspring = self.ro.recombine(p1,p2)
				new_alpha = p1.alpha + 2 * (rnd.random() - 0.5) * (p1.alpha - p2.alpha)
				self.offsprings.append(Individual(perm_offspring, self.objf(perm_offspring), new_alpha))

	def mutate(self):
		"""
		Mutate the population.
		"""
		for ind in self.offsprings:
			if self.mo.mutate(ind):
				ind.fitness = self.objf(ind.perm)

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
		for ind in self.offsprings + self.population:
			self.lso.improve(ind)

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
		slope_progress = self.tsp.cc.get_slope_progress()
		#print("slope_progress = " + str(slope_progress))
		#print("slope = " + str(self.tsp.cc.slope))
		self.so.update(slope_progress)
		self.mo.update(best_obj, worst_obj, slope_progress)
		self.lso.update(best_obj, worst_obj, slope_progress)

		self.mutate()
		self.create_offsprings()
		self.local_search()
		self.elimination()


class Individual:
	"""
	A class that represents an order in which to visit the cities.
	This class will represent an individual in the population.
	"""

	def __init__(self, perm, fitness, alpha):
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
		self.alpha = alpha

	def get_next(self, city):
		"""
		Get the city that follows next to the given city
		:param city: The number of the city (starting from zero)
		:return: The number of the city that follows the given city in this Individual.
		"""
		return self.next_cities[city]

class KTournamentSelectionOperator:
	"""
	Class that represents a selection operator for a genetic algorithm.
	This operator is updated to make sure that we have large values for k in the beginning and small values near the end
	"""

	def __init__(self, max_k, min_k):
		"""
		Create a new KTournamentSelectionOperator object
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
		selected = rnd.sample(population, k=self.k)

		selected = sorted(selected, key=lambda ind: ind.fitness)
		return selected[0]

class InversionMutationOperator:
	"""
	Class that represents a mutation operator for a genetic algorithm.
	This operator is updated to make sure that we have large values for alpha in the beginning and small values near the end.
	"""

	def __init__(self, max_alpha, min_alpha):
		"""
		Create a new KTournamentSelectionOperator.
		:param max_alpha: The maximal value for parameter alpha.
		:param min_alpha: The minimal value for parameter alpha.
		"""
		self.max_alpha = max_alpha
		self.min_alpha = min_alpha
		self.alpha = 0
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
		self.harshness = - 2 * np.sqrt(-(slope_progress - 0.5) ** 2 + 0.25) + 1
		self.alpha = max(self.min_alpha, min(self.max_alpha, self.min_alpha + self.harshness * (self.max_alpha - self.min_alpha)))

	def mutate(self, ind):
		"""
		Mutate the given Individual.
		:param ind: The Individual to mutate.
		:return: True if the given Individual was mutated, False otherwise.
		"""
		#s = 1
		#if self.worst_obj != self.best_obj:
		#	s = abs(ind.fitness - self.best_obj) / abs(self.worst_obj - self.best_obj)

		#s = max(s ** (1 - self.harshness), self.harshness)
		#alpha = max(self.min_alpha, min(self.max_alpha, self.min_alpha + s * (self.max_alpha - self.min_alpha)))

		if rnd.random() <= self.alpha:
			(start, end) = random_ind(ind.perm.shape[0])
			ind.perm[start:end] = np.flip(ind.perm[start:end])
			return True
		return False

class HeuristicCrossoverOperator:
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

class KOptLocalSearchOperator:
	""" Class that represents a local search operator. """

	def __init__(self, objf, k, min_nbh, max_nbh, distance_matrix):
		"""
		Create new KOptLocalSearchOperator.
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
		Update this KOptLocalSearchOperator
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
		#s = 1
		#if self.worst_obj != self.best_obj:
		#	s = abs(ind.fitness - self.best_obj) / abs(self.worst_obj - self.best_obj)
		#nbh = int(round(self.min_nbh + s * (self.max_nbh - self.min_nbh)))
		fitness = ind.fitness
		tries = 0
		(i,j) = (0,0)
		while fitness >= ind.fitness and tries < self.nbh:
			(i,j) = random_ind(ind.n)
			fitness = self._calc_fitness_swap(i, j, ind.perm, ind.fitness)
			tries += 1

		if fitness < ind.fitness:
			ind.perm[(i+1):(j+1)] = np.flip(ind.perm[(i+1):(j+1)])
			ind.fitness = fitness
		#if fitness != ind.fitness:
		#	raise RuntimeError("{} != {} ({},{})".format(fitness, ind.fitness, i, j))


	def __improve(self, ind):
		"""
		Improve the given Individual.
		:param ind: The Individual to improve.
		"""
		#s = 1
		#if self.worst_obj != self.best_obj:
		#	s = abs(ind.fitness - self.best_obj) / abs(self.worst_obj - self.best_obj)
		#nbh = int(round(self.min_nbh + s * (self.max_nbh - self.min_nbh)))
		(best_swap, best_fitness) = self._get_best_neighbour(ind, self.nbh)
		if not best_swap is None:
			ind.perm[(best_swap[0]+1):(best_swap[1]+1)] = np.flip(ind.perm[(best_swap[0]+1):(best_swap[1]+1)])
			ind.fitness = best_fitness

	def _improve(self, ind, k=2):
		if k == 0:
			return

		nbs_t = self._get_neighbours(ind)
		nbs = [Individual(flip_copy(ind.perm, nb_t[0][0], nb_t[0][1]), nb_t[1], ind.alpha) for nb_t in nbs_t]
		for i in nbs:
			self._improve(i, k-1)

		best_nb = min(nbs, key= lambda indi: indi.fitness)
		if best_nb.fitness < ind.fitness:
			ind.fitness = best_nb.fitness
			ind.perm = best_nb.perm


	def _calc_fitness_swap(self, i, j, perm, fitness):
		"""
		Calculate the new fitness of given permutation if edge i and j were swapped.
		:param i: The first edge
		:param j: The second edge
		:param perm: The permutation to calculate the new fitness of
		:return: The new fitness of given permutation if edge i and j were swapped.
		"""

		for k in range(i, j+1):
			fitness -= self.distance_matrix[perm[k]][perm[(k + 1) % self.n]]
		#fitness -= np.sum(np.array([self.distance_matrix[perm[k]][perm[(k + 1) % self.n]] for k in range(i, j+1)]))
		fitness += self.distance_matrix[perm[(i + 1) % self.n]][perm[(j + 1) % self.n]]
		fitness += self.distance_matrix[perm[i]][perm[j]]
		#fitness += np.sum(np.array([self.distance_matrix[perm[(k+1) % self.n]][perm[k]] for k in range(i+1, j)]))

		for k in range(i+1, j):
			fitness += self.distance_matrix[perm[(k+1) % self.n]][perm[k]]
		return fitness

	def _get_neighbours(self, ind):
		"""
		Get the neighbours of the given Individual.
		:param ind: The Individual to calculate the neighbours for.
		:return: A list of ((i,j), f) where (i,j) is the swap and f the fitness.
		"""
		swaps = [random_ind(ind.n) for _ in range(self.nbh)]
		nbs = []
		for i, j in swaps:
			fitness = self._calc_fitness_swap(i,j,ind.perm, ind.fitness)
			nbs.append(((i,j),fitness))

		return nbs

	def _get_best_neighbour(self, ind, nbh):
		swaps = [random_ind(ind.n) for _ in range(nbh)]
		best_fitness = ind.fitness
		best_swap = None
		for i, j in swaps:
			fitness = self._calc_fitness_swap(i,j,ind.perm, ind.fitness)
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
		self.min_slope = float('inf')
		self.best_objs = []
		self.slope = 0
		self.weight = weight

	def update(self, best_obj):
		"""
		Update this ConvergenceChecker
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

class EliminationOperator:
	""" Class that represents an elimination operator. """

	def __init__(self, keep, k, elite, df):
		"""
		Create a new EliminationOperator.
		:param keep:
		:param k: The amount of individuals to sample for choosing the victim.
		:param elite: The amount of individuals that go on to the next generation without doubt
		:param df: The distance function to use.
		"""
		self.keep = keep
		self.k = k
		self.elite = elite
		self.distance = df

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
			survivor = sorted_offsprings.pop(0)
			new_population.append(survivor)
			victims = rnd.sample(sorted_offsprings, k=min(self.k, len(sorted_offsprings)))
			sorted_offsprings.remove(min(victims, key= lambda ind: self.distance(ind.perm, survivor.perm)))
		return new_population

class RoundRobinEliminationOperator:
	""" Class that represents another elimination operator. """

	def __init__(self, keep, q):
		"""
		Create a new RoundRobinEliminationOperator.
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
			opponents = rnd.sample(pop, k=self.q)
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

def common_edges(a, b):
	n = a.shape[0]
	edges = set()
	for i in range(n):
		edges.add(frozenset([ a[i], a[(i+1) % n] ]))
		edges.add(frozenset([ b[i], b[(i+1) % n] ]))

	return 2 * n - len(edges)