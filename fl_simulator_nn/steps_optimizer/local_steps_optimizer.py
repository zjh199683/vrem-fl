import numpy as np

from typing import Tuple


class MobilityStepsOptimizer:

    def __init__(
            self,
            model_size: int,
            comp_slots_min: int,
            comp_slots_max: int,
            time_slot: float,
            tx_strategy: str,
            l1: float = 1.,
            rho1: float = 1.,
            rho2: float = 2e-2,
            w_latency: float = 1.,
            w_tx_time: float = 10.
    ):

        self.model_size = model_size
        self.comp_slots_min = comp_slots_min
        self.comp_slots_max = comp_slots_max
        self.time_slot = time_slot
        self.tx_strategy = tx_strategy
        self.w_latency = w_latency
        self.w_tx_time = w_tx_time
        self.conv_proxy = (lambda
                           loss_init, norm_grad_init, comp_slots, comp_slots_target:
                           norm_grad_init * ((1 - l1) ** (comp_slots - 1))
                           + rho1 * comp_slots / norm_grad_init
                           + rho2 * (comp_slots - comp_slots_target) ** 2)

    def optimize(
            self,
            comp_steps_init: int,
            max_latency: float,
            time_now: int,
            time_start: int,
            bitrate: list
    ) -> Tuple[float, int, int, int]:

        """
        Optimize computation and communication steps for a given client.

        Parameters
        ----------
        comp_steps_init: int
            initial number of computation steps
        max_latency: float
            maximum allowed latency
        time_now: int
            current simulation time
        time_start: int
            time when the client started the computation
        bitrate: list
            list of bitrates for each time slot

        Returns
        -------
        float
            cost of the optimization problem
        int
            number of computation steps
        int
            number of communication steps
        int
            number of idle steps

        """
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
                while tx_bits < self.model_size and (
                        tx_slot_last - slots_offset) * self.time_slot <= max_latency - self.time_slot:
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

        """
        Adjust local computation steps for a given client.

        Parameters
        ----------
        loss_init: float
            initial loss value
        norm_grad_init: float
            initial gradient norm
        idle_slots: int
            number of idle slots
        comp_slots_init: int
            initial number of computation steps
        comp_slots_target: int
            server target number of computation steps
        batch_size: int
            number of updates performed in one time slot

        Returns
        -------
        int
            adjusted number of local computation steps, including the batch size (i.e., the number of updates done
            in one time slot)

        """

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
