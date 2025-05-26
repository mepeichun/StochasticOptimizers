import numpy as np
from matplotlib import pyplot as plt
from stochastic_optimizers import GeneticAlgorithm, CrossEntropyOptimizer


# reference:   https://en.wikipedia.org/wiki/Test_functions_for_optimization
def rastrigin_func(num_variable=10, a=10):
    def build_func(x):
        return num_variable * a + np.sum(x ** 2 + a * np.cos(2 * np.pi * x), axis=1)
    return build_func

def sphere_func():
    def build_func(x):
        return np.sum(x ** 2, axis=1)
    return build_func

def rosenbrock_func(num_variable=10):
    def build_func(x):
        value = 0
        for i in range(num_variable - 1):
            value += 100 * (x[:, i+1] - x[:, i] ** 2) ** 2 + (1 - x[:, i]) ** 2
        return value
    return build_func


if __name__ == '__main__':
    # Example usage
    obj_func = rosenbrock_func() # Change to rastrigin_func() or sphere_func() as needed

    # Minimize using Genetic Algorithm
    ga_opt = GeneticAlgorithm(obj_func, x_min=-1, x_max=1, num_samples=1000, max_iter=1000)
    opt_x_ga, history_x_ga, history_obj_ga = ga_opt.optimize(return_history=True)
    print("The output solution is {} with an objective value of {}.".format(np.round(opt_x_ga, 2), obj_func(opt_x_ga.reshape(1, -1))[0]))

    # Minimize using Cross-Entropy Method
    ce_opt = CrossEntropyOptimizer(obj_func, num_variables=10, num_samples=1000, max_iter=1000)
    opt_x_ce, history_x_ce, history_obj_ce = ce_opt.optimize(return_history=True)
    print("The output solution is {} with an objective value of {}.".format(np.round(opt_x_ce, 2), obj_func(opt_x_ce.reshape(1, -1))[0]))

    # Plot and compare the optimization history
    plt.plot(history_obj_ga, label='Genetic Algorithm', color='blue')
    plt.plot(history_obj_ce, label='Cross-Entropy Method', color='orange')
    plt.title('Convergence of Optimization Algorithms')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.show()
