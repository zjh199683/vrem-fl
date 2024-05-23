import cvxpy as cp
import numpy as np


def find_local_minimum(x, t):
    """ Finds the first local minimum in an array, starting from index t. """
    for i in range(t, len(x)):
        try:
            if x[i+1] > x[i]:
                break
        except IndexError:
            pass
    return i


def global_problem(T, M, constants, free_buffer_evolution, x_min=0):
    """ Solves the original problem directly. """
    A = constants[0]
    B = constants[1]
    C = constants[2]
    x = cp.Variable((M, T), "steps")
    constraints = [x >= x_min]
    max_term = cp.max(cp.power(x, 2) - x, axis=0)
    objective = A * cp.sum(cp.power(x, -1))
    objective += B * cp.sum(x)
    objective += C * cp.sum(max_term)
    for t in range(T):
        for m in range(M):
            constraints.append(cp.sum(x[m, :t]) <= free_buffer_evolution[m, t])
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve("GUROBI")
    return x.value


def distributed_algorithm(T, M, constants, free_buffer_evolution, x_min=0):
    """ Implements the distributed solution to solve the algorithm. """
    A = constants[0]
    B = constants[1]
    C = constants[2]
    avg_buffer = np.zeros((M, T))
    for t in range(T):
        for m in range(M):
            avg_buffer[m, t] = free_buffer_evolution[m, t] / (t+1)
    # optimization
    t0 = 0
    t_min = np.zeros(M, dtype=int)
    solution = np.zeros((M, T))
    second_solution = [list() for _ in range(M)]
    while t0 < T:
        max_m = None
        for m in range(M):
            t_min[m] = find_local_minimum(avg_buffer[m], t0) + 1
            x = cp.Variable(int(t_min[m]-t0))
            constraints = [x >= x_min, cp.sum(x) <= t_min[m] * avg_buffer[m, t_min[m]-1]]
            objective = A * cp.sum(cp.power(x, -1))
            objective += B * cp.sum(x)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve("GUROBI")
            second_solution[m] = x.value
            objective += C * cp.sum(cp.power(x, 2) - x)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve("GUROBI")
            first_solution = x.value
            try:
                if first_solution[0] > max_m:
                    max_m = first_solution[0]
            except TypeError:
                max_m = first_solution[0]
        min_t = np.min(t_min)
        temp_solution = np.zeros((M, min_t-t0), dtype=float)
        for m in range(M):
            temp_solution[m] = second_solution[m][:min_t-t0]
        solution[:, t0:min_t] = np.clip(temp_solution, 0, max_m)
        t0 = min_t

    return solution


def second_distributed_algorithm(T, M, constants, free_buffer_evolution, x_min=0):

    A = constants[0]
    B = constants[1]
    C = constants[2]
    avg_buffer = np.zeros((M, T))
    for t in range(T):
        for m in range(M):
            avg_buffer[m, t] = free_buffer_evolution[m, t] / (t + 1)
    # optimization
    t0 = [0] * M
    t_min = np.zeros(M, dtype=int)
    solution = np.zeros((M, T))
    first_solution = np.zeros((M, T))
    while min(t0) < T:
        for m in range(M):
            if t0[m] >= T:
                continue
            t_min[m] = find_local_minimum(avg_buffer[m], t0[m]) + 1
            x = cp.Variable(int(t_min[m] - t0[m]))
            constraints = [x >= x_min, cp.sum(x) <= t_min[m] * avg_buffer[m, t_min[m] - 1]]
            objective = A * cp.sum(cp.power(x, -1))
            objective += B * cp.sum(x)
            objective += C * cp.sum(cp.power(x, 2) - x)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve("GUROBI")
            first_solution[m, t0[m]:t0[m]+len(x.value)] = x.value
            t0[m] += x.shape[0]
    max_steps = np.max(first_solution, axis=0)
    t0 = [0] * M
    t_min = np.zeros(M, dtype=int)
    while min(t0) < T:
        for m in range(M):
            if t0[m] >= T:
                continue
            t_min[m] = find_local_minimum(avg_buffer[m], t0[m]) + 1
            x = cp.Variable(int(t_min[m] - t0[m]))
            constraints = [x >= x_min,
                           cp.sum(x) <= t_min[m] * avg_buffer[m, t_min[m] - 1],
                           x <= max_steps[t0[m]:t0[m]+x.shape[0]]]
            objective = A * cp.sum(cp.power(x, -1))
            objective += B * cp.sum(x)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve("GUROBI")
            solution[m, t0[m]:t0[m] + len(x.value)] = x.value
            t0[m] += x.shape[0]

    return solution


if __name__ == "__main__":

    np.random.seed(0)
    M = 8
    T = 15
    battery = np.zeros((M, T))
    avg_energy = np.random.uniform(5, 15, M)
    battery[:, 0] = np.random.normal(avg_energy, 1)
    for t in range(1, T):
        battery[:, t] = battery[:, t-1] + np.random.normal(avg_energy, 1)
    constants = [100, 1, 0.1]
    global_solution = global_problem(T, M, constants, battery)
    import time
    a1 = time.time()
    distributed_sol_1 = distributed_algorithm(T, M, constants, battery)
    a2 = time.time()
    distributed_solution = second_distributed_algorithm(T, M, constants, battery)
    a3 = time.time()
    print("Time first algorithm:", a2-a1)
    print("Time second algorithm:", a3-a2)
    print(global_solution - distributed_solution)



