import copy
import numpy as np

from typing import Dict, Tuple, Union, List

from scheduling_algos.scheduler import Scheduler
from scheduling_algos.local_optimizer import LocalOptimizer


def loss(
        X: np.ndarray, 
        Y: np.ndarray, 
        theta_k: np.ndarray, 
        LNR: float = 1e-10
    ) -> float:
    N = X.shape[1]
    c = np.dot(X.T, theta_k) - Y
    
    return (1 / (2 * N)) * np.dot(c, c) + (LNR / 2) * np.dot(theta_k.T, theta_k)


def grad(
        X: np.ndarray, 
        Y: np.ndarray, 
        theta_k: np.ndarray, 
        LNR: float = 1e-10
    ) -> np.ndarray:
    N = X.shape[1]
    XXT = np.dot(X, X.T)
    c = np.dot(X, Y)

    return (1 / N) * (np.dot(XXT, theta_k) - c.squeeze()) + LNR * theta_k


def condAndOptSS(
        X: np.ndarray,
        LNR: float = 1e-10
    ) -> Tuple[float, float]:
    # the following is the Hessian for LSs
    n = X.shape[0]
    N = X.shape[1]
    H = (1 / N) * np.dot(X, X.T) + LNR * np.eye(n)
    lambdas = np.linalg.eigvalsh(H)
    optSS = 2 / (lambdas[-1] + lambdas[0])
    condNumb = lambdas[-1] / lambdas[0]
    
    return condNumb, optSS


def GD_LS(
        iterations: int,
        X: np.ndarray, 
        Y: np.ndarray, 
        theta_k: np.ndarray = None,
        thetaStar: np.ndarray = None, 
        LNR: float = 1e-10, 
        AGD: bool = False, 
        doBacktracking: bool = False,
        returnParam: bool = False, 
        stepSize: float = 0.01, 
        verbose: bool = False
    ) -> Union[Tuple[List[float], List[float]], List[float], np.ndarray]:
    n = X.shape[0]
    N = X.shape[1]
    costs = []
    distances = []
    thetas = []
    if AGD:
        v_GD = np.zeros((n, 1))
        myBetaMom = 0.5

    if theta_k is None:
        theta_k = np.zeros((n,))  # initial condition

    for k in range(iterations):
        if verbose:
            if np.mod(k, 20) == 0:
                print(k)
        thetas.append(copy.deepcopy(theta_k))
        g = grad(X, Y, theta_k, LNR)

        if AGD:
            v_GD = myBetaMom * np.squeeze(v_GD) + g

        cost = loss(X, Y, theta_k, LNR)
        costs.append(cost)
        if doBacktracking:
            if AGD:
                winningStep = armijoBacktracking(
                    theta_k, [X], [Y], N, v_GD, g, cost, LNR, stabilityConst=1e-32,
                    alpha=0.25, beta=0.5, costFunc='LinReg', myHint=4)
                theta_k = theta_k - winningStep * v_GD
            else:
                winningStep = armijoBacktracking(
                    theta_k, [X], [Y], N, g, g, cost, LNR, stabilityConst=1e-32,
                    alpha=0.25, beta=0.5, costFunc='LinReg', myHint=4)
                theta_k = theta_k - winningStep * g
        else:
            theta_k = theta_k - stepSize * g

    if not (thetaStar is None):
        for thisRound in range(iterations):
            distances.append(np.linalg.norm(thetas[thisRound] - thetaStar.squeeze()))
        return costs, distances
    elif returnParam:
        return theta_k  # return last version of param
    
    return costs


def Fed_GD_LS(
        X: np.ndarray, 
        Y: np.ndarray, 
        Ds: Dict[int, np.ndarray], 
        Ys: Dict[int, np.ndarray], 
        model_size: int,
        rounds: int, 
        client_data: Dict[int, np.ndarray], 
        time_slot: float, 
        m: int,
        max_latency: float,
        mobility: bool = True, 
        comp: str = 'opt', 
        scheduling: str = 'optimal', 
        batch_size: int = 1, 
        comp_slots_min: int = 1,
        tx: str = 'opt', 
        reg_lambda: float = 1e-6, 
        aoi_only: bool = False,
        beta: float = 1e-3
    ) -> Tuple[List[float], List[float], List[int], List[float], List[float]]:

    n = X.shape[0]
    # initialize theta_k and thetas to store the evolution of theta_k
    w = np.zeros((n,))
    thetas = []
    # initialize costs, distances and accuracies
    costs = []
    distancesFromOptimum = []

    thetaStar, _ = solveWithNewton(X, Y, LNR=reg_lambda)

    # create scheduler
    scheduler = Scheduler(beta=beta)
    if aoi_only:
        scheduler.set_priority_params(alpha=0)

    # create clients
    optimizers, optSSs, condNumbs, time_start, time_end = {}, {}, {}, {}, {}
    for client in client_data:
        scheduler.add_client(client)
        time_start[client] = client_data[client][0]  # first time instant when client is available
        time_end[client] = time_start[client] + len(
           client_data[client][-1]['bitrate']) - 1  # last time instant when client is available
        # create local optimization blocks
        try:
            btr = client_data[client][-1]['estimBitrate'] / 5e4
        except KeyError:
            btr = client_data[client][-1]['bitrate'] / 5e4

        optimizers[client] = LocalOptimizer(
           model_size=model_size, 
           comp_slots_min=comp_slots_min,
           time_start=time_start[client], 
           time_slot=time_slot, 
           bitrate=btr
        )
        condNumb, optStepSize = condAndOptSS(Ds[client], LNR=reg_lambda)
        optSSs[client] = optStepSize
        condNumbs[client] = condNumb

    time_now = min(time_start.values())  # track current time for mobility ~ channel quality estimation
    if mobility:
        available_clients = set()  # IDs of clients available at current time
        queued_clients = set(client_data.keys())  # IDs of clients that are still to begin
        
    else:
        available_clients = set(client_data.keys())

    slots = []
    steps = []
    tx_steps = []
    for thisRound in range(rounds):
        print("Round", thisRound)
        thetas.append(copy.deepcopy(w))
        cost = loss(X, Y, w, reg_lambda)
        costs.append(cost)

        if mobility:
            # update available clients
            available_clients = set([client for client in available_clients.union(queued_clients) if
                                    (time_end[client] >= time_now >= time_start[client])])
            queued_clients = queued_clients.difference(available_clients)

            if len(available_clients.union(queued_clients)) == 0:
                break
            
            elif len(available_clients) == 0:
                time_now += time_slot
                continue

        # scheduling
        local_steps_centr = scheduler.target_computation(m=m)  # computation steps from centralized convergence rate
        latencies_all = dict()
        time_score = dict()
        local_steps_all = dict()
        slots_pre_tx_all = dict()
        tx_time = dict()
        print("Number of available clients:", len(available_clients))
        for client in available_clients:
            norm_grad_k_client = np.linalg.norm(grad(Ds[client], Ys[client], w, reg_lambda))
            if mobility:
                latency_client, score_client, comp_slots_client, tx_steps_client, idle_slots_client = optimizers[client].local_optimization(
                    norm_grad_x0=norm_grad_k_client,
                    h_opt=local_steps_centr,
                    max_latency=max_latency,
                    time_now=time_now,
                    condNumb=condNumbs[client],
                    comp=comp,
                    tx=tx,
                    batch_size=batch_size
                )
                latencies_all[client] = latency_client
                time_score[client] = score_client
                local_steps_all[client] = comp_slots_client * int(batch_size * time_slot)
                slots_pre_tx_all[client] = comp_slots_client + idle_slots_client
                tx_time[client] = tx_steps_client
                
            else:
                local_steps_all[client] = int(batch_size * time_slot)
                latencies_all[client] = local_steps_all[client]

        # indices of scheduled agents
        if scheduling == 'all':
            scheduled_clients = list(copy.copy(available_clients))
        elif scheduling == 'random':
            scheduled_clients = scheduler.random_schedule(available_clients)
        elif scheduling == 'round_robin':
            scheduled_clients = scheduler.round_robin_schedule(available_clients)
        elif scheduling == 'optimal':
            time_feature = time_score
            if tx == 'min_tx_time':
                time_feature = tx_time

            scheduled_clients = scheduler.schedule(time_feature)
        else:
            raise NotImplementedError("Scheduling strategy requested not implemented.\n "
                                      "Choose one among:\n 'all' \n 'random' \n 'round_robin' \n 'optimal'")

        w_new = np.zeros((n,))  # global model initialization

        # compute actual round latency
        latency_round = 0
        latencies_all_true = dict()
        iter_clients = scheduled_clients.copy()
        strugglers = []
        tx_time_all_true = dict()
        for client in iter_clients:
            tx_bits = 0
            slots_offset_client = int((time_now - time_start[client]) / time_slot)
            tx_slot_last_client = slots_offset_client + slots_pre_tx_all[client]
            while tx_bits < model_size and (tx_slot_last_client - slots_offset_client) <= max_latency / time_slot:
                try:
                    tx_bits += client_data[client][1]['bitrate'][tx_slot_last_client] / 5e4 * time_slot
                except IndexError:
                    break
                
                tx_slot_last_client += 1
                
            if tx_bits >= model_size:
                latency_client_true = (tx_slot_last_client - slots_offset_client) * time_slot
                latencies_all_true[client] = latency_client_true
                tx_time_all_true[client] = (tx_slot_last_client - slots_offset_client -
                                            slots_pre_tx_all[client]) * time_slot
                if latency_round < latency_client_true:
                    latency_round = latency_client_true
                    
            else:
                scheduled_clients.remove(client)
                
                # if the client cannot transmit in time, 
                # the scheduler will wait till the deadline expires
                latency_round = max_latency
                strugglers.append(client)
                
        if len(strugglers):
            print('Strugglers:', strugglers)
            print('Latency gap:', max_latency - latencies_all[strugglers[0]])
            
        # updates by scheduled clients
        print("Number of scheduled clients:", len(scheduled_clients))
        total_steps = 0
        total_tx_steps = 0

        for client in scheduled_clients:
            print(f'local steps of client {client}: {local_steps_all[client]}')
            print(f'estimated latency for client {client}: {latencies_all[client]}')
            print(f'actual latency for client {client}: {latencies_all_true[client]}')
            print(f'est tx time for client {client}: {tx_time[client]}')
            total_steps += local_steps_all[client]
            total_tx_steps += tx_time_all_true[client]
            w_client = GD_LS(
                iterations=local_steps_all[client], 
                X=Ds[client], 
                Y=Ys[client], 
                theta_k=w, 
                LNR=reg_lambda,
                doBacktracking=False, 
                returnParam=True, 
                stepSize=optSSs[client], 
                verbose=False
            )
            w_new = w_new.squeeze() + w_client.squeeze()

        try:
            steps.append(total_steps / len(scheduled_clients))
            tx_steps.append(total_tx_steps / len(scheduled_clients))
            w_old = w.copy()
            w = w_new / len(scheduled_clients)
            print("Parameter difference (norm):", np.linalg.norm(w_old - w))
            time_now += np.ceil(latency_round / time_slot)
            slots.append(time_now)
            if time_now >= 3600:
                break
            print('time:', time_now)
        except ZeroDivisionError:
            break

    # now compute evolution of distance from optimum:
    for thisRound in range(len(thetas)):
        distancesFromOptimum.append(np.linalg.norm(thetas[thisRound] - thetaStar))

    return costs, distancesFromOptimum, slots, steps, tx_steps


def solveWithNewton(
        X: np.ndarray, 
        Y: np.ndarray, 
        LNR: float = 1e-6
    ) -> Tuple[np.ndarray, float]:
    n = X.shape[0]
    N = X.shape[1]
    theta_k = np.zeros((n,))
    H = (1 / N) * np.dot(X, X.T) + LNR * np.eye(n)
    invH = np.linalg.inv(H)
    print("Cond number:")
    print(np.linalg.cond((np.dot(X, X.T))))
    g = grad(X, Y, theta_k, LNR)
    p = np.dot(invH, g)
    theta_k = theta_k - p
    optCost = loss(X, Y, theta_k, LNR)

    return theta_k, optCost


def armijoBacktracking(
        theta_k: np.ndarray, 
        Ds: List[np.ndarray], 
        Ys: List[np.ndarray], 
        N: int, 
        pDecFib: np.ndarray, 
        g: np.ndarray, 
        costFib: float, 
        reg_lambda: float, 
        stabilityConst: float = 1e-32, 
        alpha: float = 0.25, 
        beta: float = 0.5,
        costFunc: str = 'LogReg', 
        myHint: int = 1
    ) -> float:
    csum = np.cumsum(np.ones((50,)))
    csum = np.hstack((0, csum))
    A_LRs = myHint * np.power(beta, csum)
    Losses = np.zeros((len(A_LRs), 1))  # for step-size tuning
    quadTerm = alpha * np.dot(g.T, pDecFib)
    step = 0
    notSat = True
    M = len(Ds)
    while ((step < len(A_LRs) - 1) and notSat):
        for agent in range(M):
            if costFunc == 'LogReg':
                Nagent = len(Ys[agent][0])
            else:
                Nagent = len(Ys[agent])
            wk = theta_k - A_LRs[step] * pDecFib.squeeze()
            if costFunc == 'LogReg':
                A_INCR_this = sigmoid(np.dot(wk.T, Ds[agent]))
                cost = (-1) * (1 / Nagent) * sum(np.squeeze(
                    np.multiply(Ys[agent].squeeze(), np.log(A_INCR_this.squeeze() + stabilityConst)) + np.multiply(
                        (1 - Ys[agent].squeeze()), np.log(1 - A_INCR_this.squeeze() + stabilityConst)))) + (
                                   (reg_lambda / 2) * np.dot(wk.T, wk))
            else:
                cost = (1 / (2 * Nagent)) * np.dot(np.dot(Ds[agent].T, wk) - Ys[agent],
                                                   np.dot(Ds[agent].T, wk) - Ys[agent]) + (reg_lambda / 2) * np.dot(
                    wk.T, wk)
            Losses[step] = Losses[step] + (Nagent / N) * cost
        if Losses[step] <= costFib - A_LRs[step] * quadTerm:
            notSat = False
        else:
            step = step + 1

    return A_LRs[step]
