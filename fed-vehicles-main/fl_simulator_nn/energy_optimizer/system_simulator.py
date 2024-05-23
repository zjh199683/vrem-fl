import warnings
import numpy as np


class BatteriesSimulator:
    """
    Simulates multiple batteries.

    Attributes
    ----------
    n_batteries

    batteries_levels: (np.array of shape (n_batteries,)) represents the current level of all batteries

    maximum_capacities:

    minimum_capacities:

    Methods
    -------
    __init__

    update

    """
    def __init__(self, batteries_levels, maximum_capacities, minimum_capacities, n_batteries=None):
        """

        :param batteries_levels:
        :param maximum_capacities:
        :param minimum_capacities:
        :param n_batteries:

        """
        self.n_batteries = n_batteries if n_batteries is not None else len(batteries_levels)
        if isinstance(batteries_levels, float):
            self.batteries_levels = batteries_levels * np.ones(n_batteries)
        else:
            self.batteries_levels = np.array(batteries_levels).squeeze()
        if isinstance(maximum_capacities, float):
            self.maximum_capacities = maximum_capacities * np.ones(n_batteries)
        else:
            self.maximum_capacities = np.array(maximum_capacities).squeeze()
        if isinstance(minimum_capacities, float):
            self.minimum_capacities = minimum_capacities * np.ones(n_batteries)
        else:
            self.minimum_capacities = np.array(minimum_capacities).squeeze()

    def update(self, consumed_energies, received_energies):
        """

        :param consumed_energies:
        :param received_energies:
        :return:
        """
        self.batteries_levels += (received_energies - consumed_energies)
        self.batteries_levels = np.minimum(self.batteries_levels, self.maximum_capacities)


class SystemSimulator:
    """
    responsible for simulating the distributed system,
     it includes simulating the evolution of harvested energy availability with time,
     tracking computation, communication times, and the server deadline as well as
     and potentially the evolution of the battery state.

    Attributes
    ----------
    clients_weights: (np.array os size n_clients) relative weights of the participating clients,
        should sum-up to 1.0

    full_harvested_energy: (np.array of shape n_clients * n_rounds) matrix representing the harvested energy available
        at each client and communication round / time step

    current_harvested_energy: (np.array of shape n_clients * window_size)

    window_size:

    n_clients: (int)

    n_rounds: (int)

    server_deadline: (float or iterable of size n_rounds) duration tolerated by the server before receiving an answer
        from the clients

    computation_times: (iterable of size n_clients)

    transmission_times: (iterable of size n_clients)

    computation_energies: (iterable of size n_clients)

    transmission_energies: (iterable of size n_clients)

    batteries_simulator: (BatteriesSimulator)

    iter: (int) tracks the evolution of the energy simulator

    Methods
    -------
    __init__

    update

    """

    def __init__(
            self,
            clients_weights,
            full_harvested_energy,
            window_size,
            server_deadline,
            computation_times,
            transmission_times,
            computation_energies,
            transmission_energies,
            batteries_simulator=None

    ):
        self.clients_weights = clients_weights
        self.full_harvested_energy = full_harvested_energy

        self.computation_times = computation_times
        self.transmission_times = transmission_times
        self.computation_energies = computation_energies
        self.transmission_energies = transmission_energies

        self.window_size = window_size

        self.n_clients = self.full_harvested_energy.shape[0]
        self.n_rounds = self.full_harvested_energy.shape[1]# - self.window_size

        self.server_deadline = server_deadline

        self.batteries_simulator = batteries_simulator

        self.current_harvested_energy =\
            self.full_harvested_energy[:, :self.window_size]

        self.iter = 0

    def update(self, local_steps):

        if self.iter <= self.n_rounds:
            self.iter += 1

            self.current_harvested_energy =\
                self.full_harvested_energy[:, self.iter:self.iter+self.window_size]

            # update batteries
            if self.batteries_simulator is not None:
                self.batteries_simulator.update(
                    consumed_energies=self.transmission_energies + self.computation_energies * local_steps,
                    received_energies=self.current_harvested_energy[:, 0].squeeze()
                )
        else:
            warnings.warn(
                f"Energy simulator has reached its horizon limit at iteration {self.iter}",
                RuntimeWarning
            )
