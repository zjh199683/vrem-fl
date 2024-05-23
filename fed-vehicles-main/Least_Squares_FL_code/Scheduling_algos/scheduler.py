import numpy as np
from typing import List


class Scheduler:

    def __init__(self,
                 clients_IDs: List[int] = [],
                 m: int = 30,
                 C1: float = 2.0,
                 C2: float = 1e-2,
                 eps: float = 1e-3,
                 H_max: int = np.inf,
                 alpha: float = 1.0,
                 beta: float = 1e-3) -> None:

        # global convergence proxy 
        self.C1 = C1
        self.C2 = C2
        self.eps = eps
        self.H_max = H_max
        self.m = m

        # global priority score
        for client_ID in clients_IDs:
            assert isinstance(client_ID, int)

        self.clients = dict.fromkeys(clients_IDs, {'age': 1, 'rounds': 0})
        self.rounds = 0
        self.alpha = alpha
        self.beta = beta
        self.rr_index = 0

    def add_client(self, client_ID: int) -> bool:
        """
        Adds client to set of schedulable clients.

        Parameters
        ----------
        client_ID: int
                ID of new client

        Returns
        -------
        can_add: bool
                True if client was not already present in the set of schedulable clients
        """
        assert isinstance(client_ID, int)
        can_add = False
        if client_ID not in self.clients:
            self.clients[client_ID] = {'age': 1, 'rounds': 0}
            can_add = True

        return can_add

    def set_convergence_params(self, C1: float, C2: float) -> None:
        self.C1 = C1
        self.C2 = C2
        self.est_rounds = lambda h, m: (self.C1 / h + self.C2 * (1 + 1 / m) * h) / self.eps

    def set_target_precision(self, eps: float) -> None:
        self.eps = eps
        self.est_rounds = lambda h, m: (self.C1 / h + self.C2 * (1 + 1 / m) * h) / self.eps

    def set_max_computation(self, H_max: int) -> None:
        self.H_max = H_max

    def set_priority_params(self, alpha: float=None, beta: float=None, gamma: float=None) -> None:
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma

    def target_computation(self, m: int) -> int:
        """
        Estimate optimal number of local computation steps.

        Parameters
        ----------
        m: int
            number of scheduled clients
            
        Returns
        -------
        h_opt: int
            estimated optimal number of local computation steps
        """
        self.m = m
        h_opt = round(np.sqrt(self.C1 / (self.C2 * (1 + 1 / m))))
        return h_opt

    def schedule(self, latencies: dict) -> list:
        """
        Schedule clients for current round.

        Parameters
        ----------
        latencies: dict
                keys are client IDs

                values are client latencies

        Returns
        -------
        scheduled: list
                client IDs of scheduled clients
        """
        priority = {}
        for client in latencies:
            if latencies[client] == np.inf or latencies[client] == None:
                priority[client] = -1

            else:
                if self.rounds and self.clients[client]['rounds']:
                    freq_client = self.clients[client]['rounds'] / self.rounds

                else:
                    freq_client = 1

                # TODO: are we sure of this? Latency is divided and age is multiplied?
                priority[client] = self.alpha / latencies[client] + \
                                   self.beta * (self.clients[client]['age'] + 1 / freq_client)

        scheduled = list(priority.items())
        scheduled.sort(key=lambda x: x[1], reverse=True)
        scheduled = list(list(zip(*scheduled))[0][:self.m])
        for client in self.clients:
            if client not in scheduled:
                self.clients[client]['age'] += 1

            else:
                self.clients[client]['age'] = 0
                self.clients[client]['rounds'] += 1

        self.rounds += 1

        return scheduled

    def random_schedule(self, clients) -> list:
        """
            Schedule clients randomly for the current round.

            :return clients: list of the IDs of the scheduled clients.
        """
        return list(np.random.choice(list(clients), min(self.m, len(clients)), replace=False))

    def round_robin_schedule(self, clients) -> list:
        """
            Schedule clients in a round-robin fashion.

            :return selected: list of the IDs of the scheduled clients.
        """
        clients = list(clients)
        clients.sort()
        selected = clients[self.rr_index:self.rr_index+self.m]
        if len(selected) < self.m:
            selected += clients[:len(selected)-self.m]
        self.rr_index += self.m
        self.rr_index = self.rr_index % len(clients)
        return selected
