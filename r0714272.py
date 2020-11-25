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

		ap = AlgorithmParameters(la=100, mu=50, init_alpha=0.05, init_beta=0.9)
		tsp = TSP(distance_matrix, ap)

		# Initialize the population
		tsp.initialize()
		while not tsp.has_converged():
			(mean_obj, best_obj, best_sol) = tsp.report_values()
			time_left = self.reporter.report(mean_obj, best_obj, best_sol)
			if time_left < 0:
				break

			tsp.update()


		return 0

class TSP:
	""" A class that represents the evolutionary algorithm used to find solutions to the Travelling Salesman Problem (TSP) """

	def __init__(self, distance_matrix, params):
		"""
		Create a new TSPAlgorithm object.
		:param distance_matrix: The distance matrix that contains the distances between all the cities.
		:param params: A AlgorithmParameters object that contains all the parameter values for executing the algorithm
		"""
		self.distance_matrix = distance_matrix
		self.params = params
		self.n = distance_matrix.shape[0]
		self.population = []
		self.offsprings = []

	def initialize(self):
		"""
		Initialize the population using random permutations and the initial values specified in the AlgorithmParameters object.
		"""
		self.population = [Individual(np.random.permutation(self.n), self.params.init_alpha, self.params.init_beta)]



	def select(self):
		"""
		Select a parent from the population for recombination.
		#TODO: implement a selection operator.
		:return: An Individual object that represents a parent.
		"""
		pass

	def create_offsprings(self):
		""" Select 2 * mu parents from the population and apply a recombination operator on them. """
		parents = [(self.select(), self.select()) for _ in range(self.params.mu)]
		offsprings = []
		for (p1,p2) in parents:
			if rnd.random() <= p1.beta:
				offsprings.append(self.recombine(p1,p2))
			if rnd.random() <= p2.beta:
				offsprings.append(self.recombine(p2,p1))

	def recombine(self, parent1, parent2):
		"""
		#TODO: Implement a recombination operator.
		:param parent1: The first parent.
		:param parent2: The second parent.
		:return: An Individual object that represents the offspring of parent1 and parent2
		"""
		pass

	def mutate(self):
		"""
		Mutate the population.
		#TODO: should the whole population be mutated?
		"""
		for ind in self.population:
			ind.mutate()

	def elimination(self):
		"""
		Eliminate certain Individuals from the population.
		#TODO: implement an elimination operator
		#TODO: implement a diversity check
		"""
		pass

	def local_search(self):
		"""
		Apply local search to optimize the population.
		#TODO: implement a local search operator
		"""
		pass

	def has_converged(self):
		"""
		Check whether the algorithm has converged and should be stopped
		:return: True if the algorithm should stop, False otherwise
		"""
		return True

	def report_values(self):
		"""
		Return a tuple containing the following:
			- the mean objective function value of the population
			- the best objective function value of the population
			- a 1D numpy array in the cycle notation containing the best solution
			  with city numbering starting from 0
		:return: A tuple (m, bo, bs) that represent the mean objective, best objective and best solution respectively
		"""
		return 0,0,0

	"""
	Update the population.
	"""
	def update(self):
		self.create_offsprings()
		self.mutate()
		self.elimination()

class Individual:
	"""
	A class that represents an order in which to visit the cities.
	This class will represent an individual in the population.
	"""

	def __init__(self, perm, alpha, beta):
		"""
		Create a new TSPTour with the specified parameters
		:param perm: The permutation, this should be a numpy.array containing the order of the cities.
		:param alpha: The probability that this individual will mutate.
		:param beta: The probability that this individual will create an offspring.
		"""
		self.perm = perm
		self.alpha = alpha
		self.beta = beta

	def mutate(self):
		"""
		Mutate this individual.
		"""
		if rnd.random() <= self.alpha:
			self.mutate0()

	def mutate0(self):
		"""
		#TODO: implement a mutation operator
		"""
		pass


class AlgorithmParameters:
	"""
	A class that contains all the information to run the genetic algorithm.
	Attributes:
		* la: The population size
		* mu: The amount of tries to create offsprings (not every try will result in offsprings)
		* init_alpha: The initial vlaue for alpha (the probability of mutation)
		* init_beta: The initial value for beta (the probability of recombination)
	"""

	def __init__(self, la, mu, init_alpha, init_beta):
		self.la = la
		self.mu = mu
		self.init_alpha = init_alpha
		self.init_beta = init_beta


