import json
import tables
import numpy as np
from scipy.optimize import lsq_linear


def estimate_objective_constants(local_steps, server_lr, clients_weights, gradients_norms, discard=0, verbose=0):
    r"""
    Estimates the constants C_0, C_1, C_2 and C_3 to fit the linear model

    .. math::
        J\left(\gamma, \local_steps_{i,t}, g_{i,t}\right) = \frac{C_0}{\gamma}  \\
            + C_1 \gamma\sum_{t=1}^T\sum_{i=1}^M \frac{p_i^2}{\local_steps_{i,t}} \\
            + C_2 \sum_{t=1}^T\sum_{i=1}^M p_i \local_steps_{i,t} \\
            + C_3 \sum_{t=1}^T \max_{i \in \mathcal{M}}\{\local_steps_{i,t}^2-\local_steps_{i,t}

    Parameters
    ----------
    local_steps: np.array of shape (n_clients, time_steps)
    server_lr: np.array of shape (time_steps, )
    clients_weights: np.array of shape (n_clients, )
    gradients_norms: np.array of shape (time_steps, )
    verbose: int, default is 0
        if 0 no verbosity, otherwise the optimization result is printed
    
    Returns
    -------
        [C_0, C_1, C_2, C_3]

    """
    N = local_steps.shape[1] - discard
    A = np.zeros((N, 4))
    b = np.zeros(N)
    local_steps = local_steps[:, discard:]
    gradients_norms = gradients_norms[discard:]
    local_steps_ratio = (clients_weights ** 2) @ (1 / local_steps)
    local_steps_avg = clients_weights @ local_steps
    local_steps_max = np.max(local_steps, 0)
    local_steps_max = local_steps_max ** 2 - local_steps_max
    for t in range(N):
        A[t, 0] = 1 / (np.sum(server_lr[:t + 1]))  # C0
        A[t, 1] = np.mean(server_lr[:t + 1]) / (t + 1) * np.sum(local_steps_ratio[:t + 1])  # C1
        A[t, 2] = np.sum(local_steps_avg[:t + 1]) / (t + 1)  # C2
        A[t, 3] = np.sum(local_steps_max[:t + 1]) / (t + 1)  # C3
        b[t] = np.sum(gradients_norms[:t + 1]) / (t + 1)

    results = lsq_linear(A, b, bounds=(0, np.inf))

    if verbose:
        # TODO: update verbosity message
        print(results)

    return results.x


def parse_metadata(metadata_path):
    """

    :param metadata_path: path to metadata file (expected to be a .json file)
    :return:
        local_steps: np.array of shape (n_clients, time_steps)
        server_lr: np.array of shape (time_steps,)
        clients_weights: np.array of shape (n_clients,)
        gradient_norms: np.array of shape (time_steps,)

    """
    with open(metadata_path, "r") as f:
        all_metadata = json.load(f)

    local_steps = []
    server_lr = []
    clients_weights = []
    gradients_norms = []

    for time_step in all_metadata:
        current_metadata = all_metadata[time_step]

        local_steps.append(current_metadata["local_steps"])
        server_lr.append(current_metadata["server_lr"])
        clients_weights.append(current_metadata["client_weights"])
        gradients_norms.append(current_metadata["gradient_norm"])

    local_steps = np.array(local_steps)
    server_lr = np.array(server_lr).squeeze()  # TODO: add assertion
    clients_weights = np.array(clients_weights).mean(axis=0)
    gradients_norms = np.array(gradients_norms)

    return local_steps.T, server_lr, clients_weights, gradients_norms


def parse_gradients(gradients_path):
    """

    :param gradients_path: path to full gradients file (expected to be a `.h5` file)
    :return:
        full_gradients: np.array of shape (tile_steps, dimension)

    """
    full_gradients_file = tables.open_file(gradients_path, mode='r')

    full_gradients_earray = \
        full_gradients_file.create_earray(
            full_gradients_file.root,
            name='full_gradients'
        )

    gradients = full_gradients_earray.read()

    full_gradients_file.close()

    return gradients.T
