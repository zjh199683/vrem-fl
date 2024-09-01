import cvxpy as cp
import numpy as np

from typing import Tuple
from abc import ABC, abstractmethod

from utils.constants import *
from utils.mpc import build_battery_model, build_mpc


def round_local_steps(x):

    fract, rounded = np.modf(x)
    return rounded + np.diff(np.floor(np.cumsum(fract, axis=1)), prepend=0, axis=1)


class StepsOptimizer(ABC):
    """
    Abstract class for local_steps' optimizer.

    Attributes
    ----------
    system_simulator: (SystemSimulator)

    n_clients: (int)

    n_rounds: (int)
    
    constants: (List) list of the 4 constants C_0, C_1, C_2 and C_3 (see paper for details)
      
    window_size: (int) size of the look-ahead window that optimizer has access to in order to make decisions.

    local_steps_variable: (cvxpy.expressions) variable representing the local_steps

    constraints: (List[cvxpy.constraints]) list of constraints

    objective_terms: (List[cvxpy.atoms]) List of CVXPY expression representing the three terms of the objective
        function

    current_local_steps: (List[n_clients]) list holding the current number of local steps that should
        be performed by each client

    Methods
    -------
    __init__

    set_bounds

    optimize

    get_server_Lr

    get_local_steps

    get_lower_bounds

    get_upper_bounds

    compute_objective_terms

    update

    """

    def __init__(
            self,
            system_simulator,
            constants
    ):

        self.system_simulator = system_simulator
        self.constants = constants

        try:

            self.has_battery = self.system_simulator.batteries_simulator is not None

            self.n_clients = self.system_simulator.n_clients
            self.n_rounds = self.system_simulator.n_rounds
            self.window_size = self.system_simulator.window_size

            self.lower_bounds = np.zeros((self.n_clients, self.window_size))
            self.upper_bounds = np.zeros((self.n_clients, self.window_size))

            self.local_steps = np.zeros((self.n_clients, self.n_rounds))
            self.current_local_steps = np.zeros(self.n_clients)

            self.local_steps_variable = cp.Variable(shape=(self.n_clients, self.window_size))
            self.objective_terms = []

            self.compute_objective_terms()

        except AttributeError:
            pass

        self.iter = 0

    @property
    def window_size(self):
        return self.__window_size

    @window_size.setter
    def window_size(self, window_size):
        self.__window_size = int(np.clip(window_size, a_min=1, a_max=self.n_rounds))

    @abstractmethod
    def set_bounds(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def get_server_lr(self):
        pass

    def get_optimal_local_steps(self):
        try:
            if self.local_steps_variable.value.ndim < 2:
                return self.local_steps_variable.value.round()
            return round_local_steps(self.local_steps_variable.value)
        except AttributeError:
            return round_local_steps(self.local_steps_variable)

    def get_lower_bounds(self):
        """
        return the minimum possible number of local steps per client

        :return:
                lower_bounds: (List[n_clients])

        """
        return self.lower_bounds

    def get_upper_bounds(self):
        """
        return the maximum possible number of local steps per client

        :return:
            upper_bounds: (List[n_clients])

        """
        return self.upper_bounds

    def compute_objective_terms(self):
        """
        generates the objective function expression

        """
        self.objective_terms.append(
            2 * np.sqrt(self.constants[1] * self.constants[0]) * cp.sum(
                self.system_simulator.clients_weights.T @ cp.power(self.local_steps_variable + EPSILON, - 1/2)
            )
        )

        self.objective_terms.append(
            self.constants[2] * cp.sum(self.system_simulator.clients_weights.T @ self.local_steps_variable)
        )

        self.objective_terms.append(
            self.constants[3] * cp.sum(cp.max(cp.power(self.local_steps_variable, 2) - self.local_steps_variable,
                                              axis=0))
        )

    def update(self, current_local_steps):
        self.current_local_steps = current_local_steps
        self.current_local_steps[self.current_local_steps <= 0] = 0

        try:
            self.local_steps[:, self.iter:self.iter+self.window_size] = self.current_local_steps
        except ValueError:
            try:
                self.local_steps[:, self.iter:self.iter + self.window_size] = \
                    self.current_local_steps[:, self.iter:self.iter + self.window_size]
            except IndexError:
                self.local_steps[:, self.iter:self.iter + self.window_size] = np.array([self.current_local_steps
                                                                        for _ in range(self.window_size-self.iter)]).T

        self.system_simulator.update(local_steps=self.current_local_steps)

        self.iter += 1



class MobilityStepsOptimizer(StepsOptimizer):
    def __init__(
            self, 
            system_simulator,
            constants,
            model_size: int, 
            comp_slots_min: int,
            comp_slots_max: int,
            time_slot: float,
            tx_strategy: str,
            l1: float=1., 
            rho1: float=1.,
            rho2: float=2e-2,
            w_latency: float=1.,
            w_tx_time: float=10.
        ):
        super().__init__(system_simulator, constants)
        self.model_size = model_size
        self.comp_slots_min = comp_slots_min
        self.comp_slots_max = comp_slots_max
        self.time_slot = time_slot
        self.tx_strategy = tx_strategy
        self.w_latency = w_latency
        self.w_tx_time = w_tx_time
        self.conv_proxy = (lambda
                           norm_grad_init, comp_slots, comp_slots_target:
                               norm_grad_init * ((1 - l1) ** (comp_slots - 1)) 
                               + rho1 * comp_slots / norm_grad_init 
                               + rho2 * (comp_slots - comp_slots_target) ** 2)
    
    def optimize(
            self,
            comp_steps_init: int,
            max_latency: float,
            time_now: int,
            time_start: int,
            bitrate
        ) -> Tuple[float, int, int, int]:
        comp_slots = int(min(comp_steps_init, self.comp_slots_max))
        
        # optimize communication
        slots_offset = int((time_now - time_start))  # initial position to read bitrate
        cost_min = np.inf
        idle_slots = np.inf
        tx_slots = -1
        tx_is_feasible = False
        w_latency = self.w_latency
        if self.tx_strategy == 'min_tx':
            w_latency = 0
            
        while not tx_is_feasible and comp_slots >= self.comp_slots_min:
            tx_slot_init = slots_offset + comp_slots
            
            # attempt to solve optimization problem over communication with fixed computation
            while (tx_slot_init - slots_offset) * self.time_slot <= max_latency:
                tx_bits = 0
                tx_slot_last = tx_slot_init
                
                # compute cost when communication starts at time_tx_init
                while tx_bits < self.model_size and (tx_slot_last - slots_offset) * self.time_slot <= max_latency - self.time_slot:
                    try:
                        tx_bits += bitrate[tx_slot_last] * self.time_slot
                    except IndexError:
                        break
                        
                    tx_slot_last += 1
                
                if tx_bits >= self.model_size:
                    cost_k = w_latency * (tx_slot_last - slots_offset) + self.w_tx_time * (tx_slot_last - tx_slot_init)
                    if cost_min > cost_k:
                        cost_min = cost_k
                        idle_slots = tx_slot_init - (slots_offset + comp_slots)
                        tx_slots = tx_slot_last - tx_slot_init
                        
                    tx_is_feasible = True
                        
                else:
                    break
                
                tx_slot_init += 1
                if self.tx_strategy == 'min_latency':
                    tx_slot_init += np.inf

                if tx_slot_init == tx_slot_last and tx_is_feasible:
                    break
                
            # if problem is infeasible, reduce computation
            if not tx_is_feasible:
                comp_slots -= 1

        return cost_min * self.time_slot, comp_slots, tx_slots, idle_slots
    
    def adjust_local_steps(
            self,
            loss_init: float,
            norm_grad_init: float,
            idle_slots: int,
            comp_slots_init: int,
            comp_slots_target: int,
            batch_size: int
        ) -> int:
        conv_proxy = lambda comp_slots: self.conv_proxy(loss_init, 
                                                        norm_grad_init, 
                                                        comp_slots,
                                                        comp_slots_target * batch_size)
        comp_slots = self.comp_slots_min
        conv_proxy_min = np.inf
        while conv_proxy_min > conv_proxy(comp_slots * batch_size) and comp_slots <= comp_slots_init + idle_slots:
            conv_proxy_min = conv_proxy(comp_slots * batch_size)
            comp_slots += 1
            
        comp_slots -= 1
        return comp_slots * batch_size

    def get_server_lr(self):
        return

    def set_bounds(self):
        return
    
class MyopicStepsOptimizer(StepsOptimizer):
    """
    Local steps optimizer only considering one step in the future.

    """
    def __init__(
            self,
            system_simulator,
            constants,
    ):

        super(MyopicStepsOptimizer, self).__init__(
            system_simulator=system_simulator,
            constants=constants
        )
        assert self.window_size == 1, "MyopicStepsOptimizer only supports window_size=1"
        self.name = 'myopic'

    def set_bounds(self):
        server_deadline = self.system_simulator.server_deadline

        computation_times = self.system_simulator.computation_times
        transmission_times = self.system_simulator.transmission_times
        computation_energies = self.system_simulator.computation_energies
        transmission_energies = self.system_simulator.transmission_energies

        harvested_energy = self.system_simulator.current_harvested_energy.squeeze()

        self.lower_bounds = np.zeros((self.n_clients, self.window_size))

        self.upper_bounds = \
            np.minimum(
                (server_deadline - transmission_times) / computation_times,
                (harvested_energy - transmission_energies) / computation_energies
            )

        self.upper_bounds = self.upper_bounds.reshape(self.n_clients, self.window_size)

    def optimize(self):
        # solve first stage problem # old
        # solve 1-step problem
        objective = self.objective_terms[0] + self.objective_terms[1] + self.objective_terms[2]
        constraints = \
            [
                self.local_steps_variable <= self.upper_bounds,
                self.local_steps_variable >= self.lower_bounds
            ]

        prob = cp.Problem(cp.Minimize(objective), constraints)

        try:
            prob.solve('GUROBI')
        except cp.SolverError:
            prob.solve()


    def get_server_lr(self):

        tau_mean = np.mean(self.local_steps[:, :self.iter], 1).round()
        tau = self.local_steps.copy()
        tau[:, self.iter:] = tau_mean[:, np.newaxis]

        p_square = self.system_simulator.clients_weights ** 2
        tau[tau <= 0] = 0.01
        return np.sqrt(self.constants[0] / (self.constants[1] * np.sum(p_square[:, np.newaxis] / tau)))


# TODO: the method get_upper_bounds must be overwritten
class HorizonStepsOptimizer(StepsOptimizer):
    """
    Local steps optimizer considering that the total energy is available from the beginning

    """
    def __init__(
            self,
            system_simulator,
            constants,
    ):

        super(HorizonStepsOptimizer, self).__init__(
            system_simulator=system_simulator,
            constants=constants
        )

        assert self.window_size == self.n_rounds,\
            f"HorizonStepsOptimizer requires perfect knowledge of the future," \
            f"  window_size ({self.window_size}) != n_rounds ({self.n_rounds})"

        self.total_energy = self.system_simulator.full_harvested_energy.sum(axis=1)
        self.name = 'horizon'

    def get_upper_bounds(self):

        global_bounds = (self.total_energy - self.window_size * self.system_simulator.transmission_energies) \
                        / self.n_rounds
        global_bounds /= self.system_simulator.computation_energies
        if self.upper_bounds is None:
            return global_bounds
        else:
            return np.minimum(self.upper_bounds, global_bounds)

    def set_bounds(self):
        server_deadline = self.system_simulator.server_deadline

        computation_times = self.system_simulator.computation_times
        transmission_times = self.system_simulator.transmission_times

        self.lower_bounds = np.zeros((self.n_clients, self.n_rounds))

        if server_deadline != np.inf:
            self.upper_bounds = (server_deadline - transmission_times) / computation_times
            self.upper_bounds = np.reshape(np.tile(self.upper_bounds, self.n_rounds), (self.n_clients, self.n_rounds))
        else:
            self.upper_bounds = None

    def optimize(self):

        # first stage problem
        global_upper_bounds = \
            (self.total_energy - self.window_size * self.system_simulator.transmission_energies)

        global_upper_bounds /= self.system_simulator.computation_energies

        objective = self.objective_terms[0] + self.objective_terms[1] + self.objective_terms[2]
        constraints = \
            [
                self.local_steps_variable >= self.lower_bounds,
                cp.sum(self.local_steps_variable, axis=1) <= global_upper_bounds
            ]

        if self.upper_bounds is not None:
            constraints.append(self.local_steps_variable <= self.upper_bounds)

        prob = cp.Problem(cp.Minimize(objective), constraints)

        try:
            prob.solve('GUROBI')
        except cp.SolverError:
            prob.solve()

        # # second stage problem
        # objective = self.objective_terms[0] + self.objective_terms[1]
        #
        # first_problem_result = self.local_steps_variable.copy().value
        #
        # for ii in range(self.window_size):
        #     constraints.append(self.local_steps_variable[:, ii] <= cp.max(first_problem_result[:, ii]))
        #
        # prob = cp.Problem(cp.Minimize(objective), constraints)
        #
        # try:
        #     prob.solve('GUROBI')
        # except cp.SolverError:
        #     prob.solve()

    def get_server_lr(self):
        p_square = self.system_simulator.clients_weights ** 2
        return np.sqrt(self.constants[0] / (self.constants[1] * np.sum(p_square[:, np.newaxis] / self.local_steps)))


class BatteryStepsOptimizer(StepsOptimizer):

    def __init__(
            self,
            system_simulator,
            constants,
    ):

        super(BatteryStepsOptimizer, self).__init__(
            system_simulator=system_simulator,
            constants=constants
        )
        assert self.has_battery, "system_simulator should have a battery!"

        self.model, b_lvl_next = build_battery_model(self.n_clients, self.system_simulator.batteries_simulator.maximum_capacities,
                                         self.system_simulator.computation_energies,
                                         self.system_simulator.transmission_energies)
        self.mpc = build_mpc(self.model, self.n_clients, self.system_simulator.clients_weights, self.window_size,
                             self.constants, self.system_simulator.current_harvested_energy, b_lvl_next,
                             self.system_simulator.batteries_simulator.maximum_capacities)

        # raise NotImplementedError("BatteryStepsOptimizer is not implemented")

        # computation_energies = self.system_simulator.computation_energies
        # Q, self.state_matrix, self.exogenous_matrix = build_mat(-np.eye(computation_energies),
        #                                                         np.eye(self.n_clients), np.eye(self.n_clients),
        #                                                         self.window_size)
        # Qg, _, _ = build_mat(-np.eye(self.n_clients), np.eye(self.n_clients),
        #                      np.eye(self.n_clients), self.window_size)
        # self.control_matrix = sparse.hstack([Q, Qg])
        self.name = 'battery'

    def set_bounds(self):
        return

    def optimize(self):

        self.mpc.set_initial_guess()
        x0 = self.mpc.x0
        self.local_steps_variable = self.mpc.make_step(x0)
        a = 1
        #print("Total energy released:", self.local_steps_variable[self.n_clients:, :])
        #self.local_steps_variable = self.local_steps_variable[:self.n_clients, :]
    # def set_bounds(self):
    #     server_deadline = self.system_simulator.server_deadline
    #
    #     computation_times = self.system_simulator.computation_times
    #     transmission_times = self.system_simulator.transmission_times
    #     transmission_energies = self.system_simulator.transmission_energies
    #
    #     # TODO: do we need squeeze? I don't know which is the shape of it
    #     harvested_energies = self.system_simulator.current_harvested_energy.squeeze()
    #
    #     self.lower_bounds = np.zeros((self.n_clients, self.window_size))
    #
    #     self.upper_bounds = (server_deadline - transmission_times) / computation_times
    #     self.upper_bounds = self.upper_bounds.reshape(self.n_clients, self.window_size)
    #     self.energy_matrix = harvested_energies - transmission_energies

    # def optimize(self):
    #     # first stage problem
    #     objective = self.objective_terms[0] + self.objective_terms[1] + self.objective_terms[2]
    #     # TODO: why are they private attributes? How can I add constraints?
    #     b_min = self.system_simulator.__minimum_capacities
    #     b_max = self.system_simulator.__maximum_capacities
    #     b_0 = self.system_simulator.__batteries_levels
    #     energy_release_variable = cp.Variable((self.n_clients, self.window_size))
    #     # binary variable for non-linear constraint
    #     y = cp.Variable((self.n_clients, self.window_size), boolean=True)
    #     # big-M constant (need to be big enough, e.g., one order bigger than b_max)
    #     M = 100
    #     global_variable = cp.vstack([self.local_steps_variable, energy_release_variable])
    #     global_constraint_variable = self.control_matrix @ global_variable
    #     fixed_contribution = self.state_matrix @ b_0 + self.exogenous_matrix @ self.H
    #     battery_state = global_constraint_variable + fixed_contribution
    #     # TODO: check if the last constraint works
    #     # probably global_constraint_variable + fixed_contribution <= b_max is useless
    #     constraints = \
    #         [
    #             self.local_steps_variable <= self.upper_bounds,
    #             self.local_steps_variable >= self.lower_bounds,
    #             battery_state <= b_max,
    #             battery_state >= b_min,
    #             energy_release_variable - battery_state + b_max >= - M * (1 - y),
    #             energy_release_variable - battery_state + b_max <= M * (1 - y),
    #             battery_state - b_max >= - M * (1 - y),
    #             energy_release_variable >= - M * y,
    #             energy_release_variable <= M * y,
    #             battery_state - b_max <= M * y
    #
    #         ]
    #
    #     prob = cp.Problem(cp.Minimize(objective), constraints)
    #
    #     try:
    #         prob.solve('GUROBI')
    #     except cp.SolverError:
    #         prob.solve()
    #
    #     # second stage problem
    #     objective = self.objective_terms[0] + self.objective_terms[1]
    #
    #     first_problem_result = self.local_steps_variable.copy().value
    #
    #     for ii in range(self.window_size):
    #         constraints.append(self.local_steps_variable[:, ii] <= cp.max(first_problem_result[:, ii]))
    #
    #     prob = cp.Problem(cp.Minimize(objective), constraints)
    #
    #     try:
    #         prob.solve('GUROBI')
    #     except cp.SolverError:
    #         prob.solve()

    def update(self, local_steps):
        # update the battery simulator with the current state
        local_steps = local_steps.squeeze()
        consumed_energy = local_steps * self.system_simulator.computation_energies \
                          + self.system_simulator.transmission_energies
        received_energy = self.system_simulator.current_harvested_energy[:, 0].squeeze()
        self.system_simulator.batteries_simulator.update(consumed_energy, received_energy)
        # update mpc state with rounded result
        self.mpc.x0['b_lvl'] = self.system_simulator.batteries_simulator.batteries_levels

    def get_upper_bounds(self):
        return

    def get_lower_bounds(self):
        pass

    def get_local_steps(self):
        pass

    def get_server_lr(self):
        # use the future local steps predicted also, not only the current
        local_steps = self.local_steps.copy()
        local_steps[:, self.iter] = self.local_steps_variable.squeeze()
        tau_mean = np.mean(local_steps[:, :self.iter + 1], 1).round()
        tau = local_steps.copy()
        tau[:, self.iter + 1:] = tau_mean[:, np.newaxis]

        p_square = self.system_simulator.clients_weights ** 2

        return np.sqrt(self.constants[0] / (self.constants[1] * np.sum(p_square[:, np.newaxis] / tau)))
