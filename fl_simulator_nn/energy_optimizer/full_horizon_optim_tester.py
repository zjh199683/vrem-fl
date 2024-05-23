import numpy as np
import cvxpy as cp


def evaluate_solution(x, constants):
    """ Returns the cost of the solution x. """
    A = constants[0]
    B = constants[1]
    C = constants[2]
    max_steps = np.max(x, axis=0)
    cost = C * np.sum(max_steps ** 2 - max_steps)
    cost += A * np.sum(1 / x)
    cost += B * np.sum(x)

    return cost


def global_problem(T, M, constants, max_energy, x_min=0):
    """ Solves the original problem directly. """
    A = constants[0]
    B = constants[1]
    C = constants[2]
    x = cp.Variable((M, T), "steps")
    constraints = [x >= x_min,
                   cp.sum(x, axis=1) <= max_energy]
    max_term = cp.max(cp.power(x, 2) - x, axis=0)
    objective = A * cp.sum(cp.power(x, -1))
    objective += B * cp.sum(x)
    objective += C * cp.sum(max_term)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve("GUROBI")
    return x.value


def distributed_algorithm(T, M, constants, max_energy, x_min=0):
    """ Implements the distributed solution to solve the algorithm. """
    A = constants[0]
    B = constants[1]
    C = constants[2]
    first_solution = np.zeros((M, T))
    solution = np.zeros((M, T))
    for m in range(M):
        x = cp.Variable(T)
        constraints = [x >= x_min,
                       cp.sum(x) <= max_energy[m]]
        objective = A * cp.sum(cp.power(x, -1))
        objective += B * cp.sum(x)
        objective += C * cp.sum(cp.power(x, 2) - x)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve("GUROBI")
        first_solution[m] = x.value
    max_steps = np.max(first_solution, axis=0)
    for m in range(M):
        x = cp.Variable(T)
        constraints = [x >= x_min,
                       cp.sum(x) <= max_energy[m],
                       x <= max_steps]
        objective = A * cp.sum(cp.power(x, -1))
        objective += B * cp.sum(x)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve("GUROBI")
        solution[m] = x.value

    return solution


if __name__ == "__main__":
    np.random.seed(0)
    M = 8
    T = 15
    tot_energy = T * np.random.uniform(5, 15, M)
    constants = [100, 1, 0.1]
    global_solution = global_problem(T, M, constants, tot_energy)
    distributed_solution = distributed_algorithm(T, M, constants, tot_energy)
    # print(global_solution - distributed_solution)
    print("Cost of global solution:", evaluate_solution(global_solution, constants))
    print("Cost of distributed solution:", evaluate_solution(distributed_solution, constants))
