import warnings
import numpy as np

from utils.constants import *
from .local_steps_optimizer import MobilityStepsOptimizer


class LocalStepsManager:
    """
    Manages  local_steps following a given strategy.
    Possible strategies are:

        (1) random: number of local local_steps and servers learning rate are set randomly
        (2) min: computation is set to the minimum allowed
        (3) opt: computation is optimized according to target function (accuracy + proximity with target computation)
        (4) max: computation steps are executed till communication starts

        If one of the strategies (3) or (4) is selected, a StepsOptimizer object must be given

    Attributes
    ----------
    strategy: possible are 'opt', 'min', 'max', and 'random'

    local_steps_optimizer: (LocalStepsOptimizer or None)

    n_clients: (int)

    min_local_steps: (int) default is 1

    max_local_steps: (int)

    rng (numpy.random._generator.Generator): default is None

    iter: (int) tracks current iteration index

    Methods
    -------
    __init__
    
    optimize_global
    
    optimize_local

    """

    def __init__(
            self,
            strategy: str,
            n_clients: int,
            client_data: dict,
            max_local_steps: int,
            min_local_steps: int = 1,
            local_steps_optimizer: MobilityStepsOptimizer = None,
            C1: float = 10.0,
            C2: float = 1e-2,
            eps: float = 1e-3,
            time_slot: float = 1
        ):
        assert min_local_steps <= max_local_steps, f"min_local_steps ({min_local_steps}) is larger" \
                                                   f" than max_local_steps ({max_local_steps})!"

        self.strategy = strategy
        self.n_clients = n_clients
        self.max_local_steps = max_local_steps
        self.min_local_steps = min_local_steps
        self.C1 = C1
        self.C2 = C2
        self.eps = eps
        self.local_steps_global_opt = None
        self.client_data = client_data
        self.comp_slots = dict.fromkeys(client_data.keys())
        self.time_score = dict.fromkeys(client_data.keys())
        self.idle_slots = dict.fromkeys(client_data.keys())
        self.tx_slots = dict.fromkeys(client_data.keys())
        self.local_steps = dict.fromkeys(client_data.keys())
        self.initial_bitrate = dict.fromkeys(client_data.keys())
        self.time_slot = time_slot

        if self.strategy in \
                [
                    'opt',
                    'max'
                ]:
            assert local_steps_optimizer is not None
            f'steps_optimizer is required with {self.strategy} strategy.'

            self.local_steps_optimizer = local_steps_optimizer

    @property
    def strategy(self):
        return self.__strategy

    @strategy.setter
    def strategy(self, strategy: str):
        if strategy in ALL_COMP_STRATEGIES:
            self.__strategy = strategy
        else:
            warnings.warn("strategy is set to random!", RuntimeWarning)
            self.__strategy = "random"


    def optimize_global(self, num_scheduled: int):
        """
        Estimate homogeneous optimal number of local computation steps.

        Parameters
        ----------
        num_scheduled: int
            maximum number of scheduled clients
            
        Returns
        -------
        h_opt: int
            estimated optimal number of local computation steps
        """
        self.local_steps_global_opt = min(
            round(np.sqrt(self.C1 / (self.C2 * (1 + 1 / num_scheduled)))),
            self.max_local_steps)


    def optimize_local(
            self,
            max_latency: int = None,
            time_now: int = None,
            available_clients: list = []
        ):
        """
        returns a sequence of number of local_steps,
         this function is expected to be called at the beginning
         of each round when used with aggregator.

        :param local_lr: value of the local learning rate, default is None
        :return:
            np.array(shape=(self.n_clients), dtype=np.uint16)

        """
        if self.strategy == "min":
            for client in self.comp_slots:
                self.comp_slots[client] = self.min_local_steps

        elif self.strategy in \
                [
                    'opt',
                    'max'
                ]:
            for client in available_clients:
                client_offset = int(time_now - self.client_data[client][0])
                score_client, comp_slots_client, tx_slots_client, idle_slots_client = self.local_steps_optimizer.optimize(
                    comp_steps_init=self.local_steps_global_opt,
                    max_latency=max_latency,
                    time_now=time_now,
                    time_start=self.client_data[client][0],
                    bitrate=self.client_data[client][1]['estimBitrate'],
                    batch_size=self.client_data[client][2]
                )
                if self.strategy == 'max':
                    comp_slots_client += idle_slots_client
                    idle_slots_client = 0

                self.comp_slots[client] = comp_slots_client
                self.idle_slots[client] = idle_slots_client
                self.tx_slots[client] = tx_slots_client
                self.initial_bitrate[client] = self.client_data[client][1]['estimBitrate'][client_offset]
                self.time_score[client] = score_client
                self.local_steps[client] = self.comp_slots[client] * self.client_data[client][2]

        else:
            error_message = f"Strategy {self.strategy} is not supported! Possible are "
            for strategy in ALL_COMP_STRATEGIES:
                error_message += f"{strategy} "

            raise NotImplementedError(error_message)


    def adjust_local_steps(
            self,
            client: int,
            loss_init: float,
            norm_grad_init: float
        ) -> int:
        local_steps_client = self.local_steps_optimizer.adjust_local_steps(
            loss_init=loss_init,
            norm_grad_init=norm_grad_init,
            idle_slots=self.idle_slots[client],
            comp_slots_init=self.comp_slots[client],
            comp_slots_target=self.local_steps_global_opt,
            batch_size=self.client_data[client][2]
        )
        self.local_steps[client] = local_steps_client
        return local_steps_client
