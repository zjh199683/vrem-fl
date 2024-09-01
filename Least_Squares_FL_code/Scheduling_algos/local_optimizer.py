import numpy as np

from typing import Tuple


class LocalOptimizer:
    
    def __init__(
            self,
            model_size: int,
            comp_slots_min: int,
            time_start: float,
            time_slot: float,
            bitrate: list,
            l1: float=1.0, 
            l2: float=1.0,
            rho1: float=1e-3,
            rho2: float=1,
            w_latency: float=1.0,
            w_tx_time: float=1.25
        ) -> None:

        self.model_size = model_size
        self.comp_slots_min = comp_slots_min
        self.time_start = time_start
        self.time_slot = time_slot
        self.bitrate = bitrate
        self.l1 = l1
        self.l2 = l2
        self.rho1 = rho1
        self.rho2 = rho2
        self.conv_proxy_val = lambda t, loss_x0, norm_grad_x0, h_opt: \
            (loss_x0 - self.l1) * ((1 - self.l2) ** (t - 1)) + self.rho1 * t / norm_grad_x0 + self.rho2 * (t - h_opt) ** 2
        self.conv_proxy_grad = lambda t, norm_grad_x0, h_opt, condNumb: \
            (norm_grad_x0) * ((1 - 1/condNumb) ** (t - 1)) + self.rho1 * t / norm_grad_x0 + self.rho2 * (t - h_opt) ** 2
        self.w_latency = w_latency
        self.w_tx_time = w_tx_time
            

    def local_optimization(
            self,
            norm_grad_x0: float,
            h_opt: int,
            max_latency: int,
            time_now: int,
            condNumb: float,
            comp: str,
            tx: str,
            batch_size: int=2
        ) -> Tuple[float, int]:
        """
        Optimize number of (batches of) computation steps and communication pattern.

        Parameters
        ----------
        comp: str
            "min": computation is set to the minimum allowed
            "opt": computation is optimized according to target function (accuracy + proximity with target computation)
            "max": computation steps are executed till communication starts
            
        Returns
        -------
        latency: float
            estimated total latency for this round 
        comp_steps: int
            number of local computation steps
        """
        slots_offset = int((time_now - self.time_start) / self.time_slot)  # initial position to read bitrate
        if not comp == 'min':
            conv_proxy = lambda t: self.conv_proxy_grad(t, norm_grad_x0, h_opt, condNumb)
            conv_proxy_min = np.Inf
            h = 1
            while conv_proxy_min > conv_proxy(h):
                conv_proxy_min = conv_proxy(h)
                h += 1
            
            comp_slots = np.ceil((h - 1) / batch_size)
            comp_slots = int(max(comp_slots, self.comp_slots_min))
            
        else:
            comp_slots = self.comp_slots_min
        
        # optimize communication
        latency = np.inf
        cost_min = np.inf
        idle_slots = np.inf
        tx_is_feasible = False
        tx_slot_init_opt = -1
        tx_time = -1
        w_latency = self.w_latency
        if tx == 'min_tx_time':
            w_latency = 0
        while not tx_is_feasible and comp_slots >= self.comp_slots_min:
            tx_slot_init = slots_offset + comp_slots
            
            # attempt to solve optimization problem over communication with fixed computation
            while (tx_slot_init - slots_offset) * self.time_slot <= max_latency:
                tx_bits = 0
                tx_slot_last = tx_slot_init
                
                # compute cost when communication starts at time_tx_init
                while tx_bits < self.model_size and (tx_slot_last - slots_offset) * self.time_slot <= max_latency - \
                        self.time_slot:
                    try:
                        tx_bits += self.bitrate[tx_slot_last] * self.time_slot
                    except IndexError:
                        break
                        
                    tx_slot_last += 1
                
                if tx_bits >= self.model_size:
                    cost_k = w_latency * (tx_slot_last - slots_offset) + self.w_tx_time * (tx_slot_last - tx_slot_init)
                    if cost_min > cost_k:
                        cost_min = cost_k
                        latency = (tx_slot_last - slots_offset) * self.time_slot
                        idle_slots = tx_slot_init - (slots_offset + comp_slots)
                        tx_slot_init_opt = tx_slot_init
                        tx_time = tx_slot_last - tx_slot_init
                        
                    tx_is_feasible = True
                        
                else:
                    break
                
                tx_slot_init += 1
                if tx == 'min_latency':
                    tx_slot_init += np.inf

                if tx_slot_init == tx_slot_last and tx_is_feasible:
                    break
                
            # if problem is infeasible, reduce computation
            if not tx_is_feasible:
                comp_slots -= 1
            
        if comp == 'max':
            if tx_is_feasible:
                comp_slots = tx_slot_init_opt - slots_offset
                idle_slots = 0

        return latency, cost_min * self.time_slot, comp_slots, tx_time, idle_slots


    def set_convergence_params(
            self, 
            l1: float, 
            l2: float
        ) -> None:
        self.l1 = l1
        self.l2 = l2


    def set_optimization_params(
            self, 
            rho1: float, 
            rho2: float
        ) -> None:
        self.rho1 = rho1
        self.rho2 = rho2


    def set_regularization_params(
            self, 
            alpha: float, 
            beta: float
        ) -> None:
        self.w_latency = alpha
        self.w_tx_time = beta

