import numpy as np
import warnings

class GeneticAlgorithm:
    def __init__(self, obj_func, x_min=-np.inf, x_max=np.inf, keep_rate=0.1,
                 num_variables=10, max_iter=1000, num_samples=100, eps=1e-6):
        """
        Genetic Algorithm optimizer.

        Parameters:
        - obj_func: Objective function to minimize.
        - x_min, x_max: Bounds for the search space.
        - keep_rate: Fraction of top solutions to retain each generation.
        - num_variables: Number of variables (dimensions).
        - max_iter: Maximum number of iteration.
        - num_samples: Population size.
        - eps: Tolerance for convergence.
        """
        self.obj_func = obj_func
        self.x_min = x_min
        self.x_max = x_max
        self.keep_rate = keep_rate
        self.num_variables = num_variables
        self.max_iter = max_iter
        self.num_samples = num_samples
        self.eps = eps
        self.scale = np.inf
        self.x_opt = None

    def crossover(self, parents, alpha=0.25):
        """
        Perform crossover between parent solutions.

        Parameters:
        - parents: Selected top individuals.
        - alpha: Blend crossover factor.

        Returns:
        - New offspring population.
        """
        num_parents = parents.shape[0]
        repeats = self.num_samples // num_parents
        father = np.tile(parents, (repeats, 1))
        mother = np.tile(parents, (repeats, 1))
        np.random.shuffle(father)
        np.random.shuffle(mother)
        prob_array = np.random.uniform(-alpha, 1 + alpha, size=father.shape)
        return father * prob_array + mother * (1 - prob_array)

    def mutate(self, offspring, mutation_prob=0.25):
        """
        Apply mutation to the offspring.

        Parameters:
        - offspring: The population to mutate.
        - mutation_prob: Probability of mutation per gene.

        Returns:
        - Mutated population.
        """
        max_val = np.max(offspring, axis=0)
        min_val = np.min(offspring, axis=0)
        std_dev = (max_val - min_val) / 6
        noise = np.random.normal(0, std_dev, size=offspring.shape)
        mutation_mask = np.random.rand(*offspring.shape) < mutation_prob
        return offspring + noise * mutation_mask

    def optimize(self, return_history=False):
        """
        Run the genetic algorithm.

        Parameters:
        - return_history: If True, returns optimization history.

        Returns:
        - Optimal solution (and optionally history).
        """
        history_x = []
        history_obj = []
        t = 0
        num_keep = int(self.num_samples * self.keep_rate)

        # Initialize population
        population = np.random.uniform(self.x_min, self.x_max, size=(self.num_samples, self.num_variables))
        population = np.clip(population, self.x_min, self.x_max)

        while t < self.max_iter and np.max(self.scale) > self.eps:
            obj_vals = self.obj_func(population)
            assert obj_vals.shape == (self.num_samples, )

            # Select top performers
            best_indices = np.argsort(obj_vals)[:num_keep]
            best_samples = population[best_indices]

            # Update current best
            self.x_opt = np.mean(best_samples, axis=0)
            self.scale = np.std(best_samples, axis=0)

            # Generate next population
            offspring = self.crossover(best_samples)
            population = self.mutate(offspring)
            population = np.clip(population, self.x_min, self.x_max)

            if return_history:
                history_x.append(self.x_opt)
                history_obj.append(self.obj_func(self.x_opt.reshape(1, -1))[0])

            t += 1

        if np.max(self.scale) > self.eps:
            warnings.warn(f"Algorithm did not converge in {self.max_iter} iterations.", RuntimeWarning)
        else:
            print(f"Algorithm converged in {t} iterations.")

        if return_history:
            return self.x_opt, history_x, history_obj
        return self.x_opt


class CrossEntropyOptimizer:
    def __init__(self, obj_func, mu=0.5, scale=1.0, num_variables=10,
                 max_iter=1000, num_samples=100, elite_frac=0.1, eps=1e-6):
        """
        Cross-Entropy optimization using Gaussian sampling.

        Parameters:
        - obj_func: Objective function to minimize.
        - mu: Initial mean.
        - scale: Initial standard deviation.
        - num_variables: Number of variables.
        - max_iter: Max number of iterations.
        - num_samples: Population size.
        - elite_frac: Fraction of top performers used to update distribution.
        - eps: Tolerance for convergence.
        """
        self.obj_func = obj_func
        self.mu = np.full(num_variables, mu)
        self.scale = np.full(num_variables, scale)
        self.num_variables = num_variables
        self.max_iter = max_iter
        self.num_samples = num_samples
        self.elite_frac = elite_frac
        self.eps = eps

    def optimize(self, return_history=False):
        """
        Run the Cross-Entropy optimization algorithm.

        Parameters:
        - return_history: If True, returns optimization history.

        Returns:
        - Optimal solution (and optionally history).
        """
        history_mu = []
        history_obj = []
        t = 0
        num_elite = int(self.num_samples * self.elite_frac)

        while t < self.max_iter and np.max(self.scale) > self.eps:
            samples = np.random.normal(self.mu, self.scale, size=(self.num_samples, self.num_variables))
            obj_vals = self.obj_func(samples)
            assert obj_vals.shape == (self.num_samples, )

            best_indices = np.argsort(obj_vals)[:num_elite]
            elite_samples = samples[best_indices]

            # Update distribution
            self.mu = np.mean(elite_samples, axis=0)
            self.scale = np.std(elite_samples, axis=0)

            if return_history:
                history_mu.append(self.mu)
                history_obj.append(self.obj_func(self.mu.reshape(1, -1))[0])

            t += 1

        if np.max(self.scale) > self.eps:
            warnings.warn(f"Algorithm did not converge in {self.max_iter} iterations.", RuntimeWarning)
        else:
            print(f"Algorithm converged in {t} iterations.")

        if return_history:
            return self.mu, history_mu, history_obj
        return self.mu
