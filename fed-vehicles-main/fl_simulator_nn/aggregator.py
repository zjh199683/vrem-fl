import os
import time
import random
import tables
import json
from metrics import *

from abc import ABC, abstractmethod


import numpy as np
from copy import deepcopy

from utils.torch_utils import *
from utils.optim import FedNovaOptimizer


class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    local_steps_manager: (LocalStepsManager)

    clients: List[Client]

    test_clients: List[Client]

    global_learner: Learner

    sampling_rate: (float) proportion of clients used at each round; default is `1.`

    sample_with_replacement: if True, client are sampled with replacement; default is False

    n_clients:

    clients_weights:

    model_dim: dimension of the used model

    c_round: index of the current communication round

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    global_train_logger:

    global_test_logger:

    gradients_save_path: path to save the full gradients (expected to be a `.h5` file);
                            if `None` gradients are not saved.

    save_gradients_flag: (bool) if `True` the gradients are saved to `__gradients_save_path`

    full_gradients_file: file to `__gradients_save_path`, it is None when `__gradients_save_path` is None

    full_gradients_earray: extendable, homogeneous datasets in an HDF5 file;
                            it is None when `__gradients_save_path` is None

    metadata_save_path: path to save full gradients norms, server learning rates and number of local local_steps;
                            expected to be a .json file

    metadata_save_flag: (bool) if `True` metadata (full gradients norms, server learning rates and local local_steps)
                            are saved

    metadata: (List[Dict]) list of dictionaries, each of them holds the full gradients norm, server learning rate
                and the number of local local_steps at a given communication round.


    rng: random number generator

    Methods
    ----------
    __init__
    mix
    update_clients
    update_test_clients
    get_full_gradients_average
    write_logs
    save_state
    load_state

    """
    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            gradients_save_path=None,
            metadata_save_path=None,
            verbose=0,
            classic_weights=False,
            seed=None,
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients
        # only used for FedNova
        self.classic_weights = classic_weights

        self.global_learner = global_learner
        self.device = self.global_learner.device
        self.model_dim = self.global_learner.model_dim

        self.log_freq = log_freq
        self.verbose = verbose
        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)

        try:
            self.clients_weights =\
                torch.tensor(
                    [client.n_train_samples for client in self.clients],
                    dtype=torch.float32
                )
            self.clients_weights = self.clients_weights / self.clients_weights.sum()
        except AttributeError:
            self.clients_weights = np.zeros(self.n_clients)

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = list()

        self.save_gradients_flag = False

        if gradients_save_path is not None:
            self.save_gradients_flag = True

            os.makedirs(os.path.split(gradients_save_path)[0], exist_ok=True)

            atom = tables.Float32Atom()
            self.__gradients_save_path = gradients_save_path
            self.full_gradients_file = tables.open_file(self.__gradients_save_path, mode='w')
            self.full_gradients_earray =\
                self.full_gradients_file.create_earray(
                    self.full_gradients_file.root,
                    name='full_gradients',
                    atom=atom,
                    shape=(self.model_dim, 0)
                )
        else:
            self.full_gradients_file = None

        self.metadata_save_path = metadata_save_path
        self.save_metadata_flag = False
        self.metadata = []

        if metadata_save_path is not None:
            self.save_metadata_flag = True
            os.makedirs(os.path.split(self.metadata_save_path)[0], exist_ok=True)

            with open(self.metadata_save_path, "w") as f:
                json.dump(dict(), f)

        self.c_round = 0

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def set_clients(self, clients, clients_weights):
        """Set the clients and reset the weights"""
        self.clients = clients
        self.n_clients = len(clients)
        self.clients_weights = torch.tensor(clients_weights / clients_weights.sum())

    def update_test_clients(self):
        for client in self.test_clients:
            copy_model(target=client.learner.model, source=self.global_learner.model)

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        if self.sample_with_replacement:
            self.sampled_clients = \
                self.rng.choices(
                    population=self.clients,
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients = self.rng.sample(self.clients, k=min(self.n_clients_per_round, len(self.clients)))

    def get_full_gradient(self):
        average_gradient = torch.zeros(self.model_dim, device=self.device)
        for client_id, client in enumerate(self.clients):
            average_gradient += self.clients_weights[client_id] * client.get_full_gradient()

        return average_gradient

    def set_lr(self, lr=None):
        # TODO: add possibility to set list of learning rates
        """
        set the learning rate for the aggregator, and returns a a list of learning rates (per params group).

        :param lr: float or None
        :return:
            List[len(self.global_learner.optimizer.param_groups)]

        """
        if lr is None:
            return

        lr_list = []
        for param_groups in self.global_learner.optimizer.param_groups:
            param_groups['lr'] = lr
            lr_list.append(float(param_groups['lr']))

        return lr_list

    def set_local_steps(self, local_steps):
        for client_id, client in enumerate(self.clients):
            client.local_steps = local_steps[client_id]

        return local_steps

    def save_metadata(self, metadata):
        if not self.save_metadata_flag:
            return

        with open(self.metadata_save_path, "r+") as f:
            data = json.load(f)
            data.update({self.c_round: metadata}, )
            f.seek(0)
            json.dump(data, f)

    def save_full_gradient(self, gradient):
        if self.save_gradients_flag:
            self.full_gradients_earray.append(gradient)
            self.full_gradients_earray.flush()

    def write_logs(self):
        self.update_test_clients()

        print('Computing logs...')
        for global_logger, clients in [
            (self.global_train_logger, self.clients),
            (self.global_test_logger, self.test_clients)
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                # train_loss, train_acc, test_loss, test_acc = client.write_logs()

                # if self.verbose > 1:
                #     print("*" * 30)
                #     print(f"Client {client_id}..")
                #
                #     print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|", end="")
                #     print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += client.train_loss * client.n_train_samples
                self.metric.confusion_matrix += client.learner.metric.confusion_matrix
                # global_train_acc += train_acc * client.n_train_samples
                # global_test_loss += test_loss * client.n_test_samples
                # global_test_acc += test_acc * client.n_test_samples

                total_n_samples += client.n_train_samples
                # total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_train_acc = self.metric.percent_mIoU() / 100
            # global_test_loss /= total_n_test_samples
            # global_train_acc /= total_n_samples
            # global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |")
                # print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            # global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            # global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        if self.verbose > 0:
            print("#" * 80)

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         as `.pt` file, and `learners_weights` for each client in `self.clients` as a single numpy array (`.np` file).

        :param dir_path:
        """
        save_path = os.path.join(dir_path, f"chkpts.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

    def load_state(self, dir_path):
        """
        load the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         from a `.pt` file, and `learners_weights` for each client in `self.clients` from numpy array (`.np` file).

        :param dir_path:
        """
        chkpts_path = os.path.join(dir_path, f"chkpts.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpts_path))


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        pass


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """
    def mix(self, manager):
        self.sample_clients()

        # assign the updated model to all clients
        #print('Updating clients')
        self.update_clients()
        #print('Clients updated and ready.')
        self.metric = Metrics([str(i) for i in range(11)])
        for client in self.sampled_clients:
            client.step(manager)
            #print('Client ' + str(client.client_id) + ' done.')

        #print('Averaging...')
        learners = [client.learner for client in self.clients]
        average_learners(learners, self.global_learner, weights=self.clients_weights)
        #print('Done.')

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for client in self.clients:

            copy_model(client.learner.model, self.global_learner.model)

            if callable(getattr(client.learner.optimizer, "set_initial_params", None)):
                client.learner.optimizer.set_initial_params(
                    self.global_learner.model.parameters()
                )


class FedNovAggregator(CentralizedAggregator):
    """
    Implements
     `Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization`__.
     (https://arxiv.org/pdf/2007.07481.pdf)

    """
    # TODO: current implementation only covers the case when local solver is Vanilla SGD
    def mix(self):

        initial_parameters = deepcopy(self.global_learner.get_param_tensor())

        for client_id, client in enumerate(self.clients):
            # TODO: needs optimization, can gather parameters while computing the average
            client.step()
            final_parameters = deepcopy(client.learner.get_param_tensor())

            cum_grad = - (final_parameters - initial_parameters) / torch.tensor(client.local_steps * client.get_lr())
            client.learner.set_grad_tensor(cum_grad)

        learners = [client.learner for client in self.clients]
        weights = self.clients_weights
        if self.classic_weights:
            clients_steps = np.array([client.local_steps for client in self.clients])
            weights = self.clients_weights * clients_steps / (self.clients_weights @ clients_steps)

        average_learners(
            learners,
            self.global_learner,
            weights=weights,
            average_gradients=True,
            average_params=False
        )

        # update parameters
        self.global_learner.optimizer_step()

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()
