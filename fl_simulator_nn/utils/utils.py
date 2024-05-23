from models import *
from datasets import *
from learner import *
from client import *
from aggregator import *
from torchmetrics import JaccardIndex
from metrics import *

from energy_optimizer.local_steps_manager import *
from energy_optimizer.system_simulator import *
from energy_optimizer.local_steps_optimizer import *

from .optim import *
from .metrics import *
from .constants import *

from torch.utils.data import DataLoader
from copy import deepcopy

from tqdm import tqdm


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment
    :param experiment_name: name of the experiment
    :return: str
    """
    data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def get_learner(
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        weight_decay,
        use_float64,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None,
        lr_state_dict=None
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param weight_decay
    :param use_float64:
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :param lr_state_dict:
    :return:
        Learner

    """
    torch.manual_seed(seed)

    if name == "synthetic":
        if output_dim == 2:
            criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
            metric = binary_accuracy
            model = LinearLayer(input_dim, 1).to(device)
            is_binary_classification = True
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
            metric = accuracy
            model = LinearLayer(input_dim, output_dim).to(device)
            is_binary_classification = False
    elif name == "cifar10":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        # model = get_mobilenet(n_classes=10).to(device)
        model = CIFAR10CNN(num_classes=10).to(device)
        is_binary_classification = False
    elif name == "cifar100":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = CIFAR10CNN(num_classes=100).to(device)
        # model = get_mobilenet(n_classes=100).to(device)
        is_binary_classification = False
    elif name == "emnist" or name == "femnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN(num_classes=62).to(device)
        is_binary_classification = False
    elif name == "shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8

        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
        metric = accuracy
        model =\
            NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"]
            ).to(device)
        is_binary_classification = False
    elif name == "apolloscape":
        criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=255).to(device)
        # metric = JaccardIndex(task="multiclass", num_classes=11, ignore_index=255).to(device)
        metric = Metrics([str(i) for i in range(11)])
        model = SegmentationCNN(pretrained=False, num_classes=11).to(device)
        is_binary_classification = False

    else:
        raise NotImplementedError

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            weight_decay=weight_decay,
            mu=mu,
            name=name
        )
    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )
    if lr_state_dict:
        lr_scheduler.load_state_dict(lr_state_dict)

    if name == "shakespeare":
        return LanguageModelingLearner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification,
            use_float64=use_float64
        )
    else:
        return Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification,
            use_float64=use_float64
        )


def get_loaders(type_, root_path, batch_size, is_validation, clients, apollo_records=None):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last
    :param type_: type of the dataset;
    :param root_path: path to the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])

    """
    if type_ == "cifar10":
        inputs, targets = get_cifar10()
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
    elif type_ == "emnist":
        inputs, targets = get_emnist()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []
    print("Clients in apolloscape:", len(clients))
    if type_ == 'apolloscape':
        for task_id in clients:

            train_iterator = \
                get_loader(
                    type_=type_,
                    path=root_path,
                    batch_size=batch_size,
                    inputs=inputs,
                    targets=targets,
                    train=True,
                    apollo_records=apollo_records[str(task_id)]
                )
            if train_iterator is None:
                print("Train iterator is None at", task_id)
                print(apollo_records[str(task_id)])
                raise ValueError("Error")
            val_iterator = \
                get_loader(
                    type_=type_,
                    path=root_path,
                    batch_size=batch_size,
                    inputs=inputs,
                    targets=targets,
                    train=False,
                    apollo_records=apollo_records[str(task_id)]
                )

            if is_validation:
                test_set = "val"
            else:
                test_set = "test"

            test_iterator = \
                get_loader(
                    type_=type_,
                    path=root_path,
                    batch_size=batch_size,
                    inputs=inputs,
                    targets=targets,
                    train=False,
                    apollo_records=apollo_records[str(task_id)]
                )

            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)

    else:
        for task_id, task_dir in enumerate(os.listdir(root_path)):
            if task_id in clients:
                task_data_path = os.path.join(root_path, task_dir)

                train_iterator = \
                    get_loader(
                        type_=type_,
                        path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                        batch_size=batch_size,
                        inputs=inputs,
                        targets=targets,
                        train=True
                    )

                val_iterator = \
                    get_loader(
                        type_=type_,
                        path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                        batch_size=batch_size,
                        inputs=inputs,
                        targets=targets,
                        train=False
                    )

                if is_validation:
                    test_set = "val"
                else:
                    test_set = "test"

                test_iterator = \
                    get_loader(
                        type_=type_,
                        path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                        batch_size=batch_size,
                        inputs=inputs,
                        targets=targets,
                        train=False
                    )

            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)

    return train_iterators, val_iterators, test_iterators


def get_loader(type_, path, batch_size, train, inputs=None, targets=None, apollo_records=None):
    """
    constructs a torch.utils.DataLoader object from the given path
    :param type_: type of the dataset; possible are `tabular`, `images` and `text`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return:
        torch.utils.DataLoader

    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "cifar10":
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets)
    elif type_ == "emnist":
        dataset = SubEMNIST(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
    elif type_ == "apolloscape":
        if apollo_records is None:
            raise TypeError("Must specify apollo_records.")
        dataset = ApolloscapeDataset(apollo_records, path)
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) <= 1:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = ((type_ == "cifar100") or (type_ == "cifar10")
                 or (type_ == "apolloscape")) and (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)


def get_client(
        learner,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        fit_epoch,
        client_id=None
):
    """

    :param learner:
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param fit_epoch

    :return:

    """
    return Client(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            fit_epoch=fit_epoch,
            client_id=client_id
    )


def get_local_steps_optimizer(cfg_file_path, model_size, tx_strategy, clients_weights=None):
    """

    :param cfg_file_path:
    :param clients_weights
    :return:
        LocalStepsOptimizer
    """

    if cfg_file_path is None:
        return

    if not os.path.exists(cfg_file_path):
        return

    with open(cfg_file_path, "r") as f:
        cfg = json.load(f)

    if clients_weights is None and not cfg['mobility']:
        warnings.warn(
            "clients weight are initialized uniformly in local steps optimizer",
            RuntimeWarning
        )

        n_clients = len(cfg["computation_times"])
        cfg["clients_weights"] = np.full(n_clients, 1/n_clients)

    else:
        cfg["clients_weights"] = deepcopy(clients_weights)

    if "batteries_maximum_capacities" in cfg:
        batteries_simulator = BatteriesSimulator(
            batteries_levels=cfg["batteries_levels"],
            maximum_capacities=cfg["batteries_maximum_capacities"],
            minimum_capacities=cfg["batteries_minimum_capacities"],
            n_batteries=len(cfg["clients_weights"])
        )

        system_simulator = SystemSimulator(
            clients_weights=cfg["clients_weights"],
            full_harvested_energy=np.array(cfg["full_harvested_energy"]),
            window_size=cfg["window_size"],
            server_deadline=cfg["server_deadline"],
            computation_times=np.array(cfg["computation_times"]),
            transmission_times=np.array(cfg["transmission_times"]),
            computation_energies=np.array(cfg["computation_energies"]),
            transmission_energies=np.array(cfg["transmission_energies"]),
            batteries_simulator=batteries_simulator
        )

        return BatteryStepsOptimizer(
            system_simulator=system_simulator,
            constants=cfg["constants"]
        )

    if cfg["mobility"]:
        return MobilityStepsOptimizer(
            system_simulator=None,
            constants=None,
            model_size=model_size,
            comp_slots_min=cfg["min_local_steps"],
            comp_slots_max=cfg["max_local_steps"],
            time_slot=cfg["time_slot"],
            tx_strategy=tx_strategy)

    system_simulator = SystemSimulator(
        clients_weights=cfg["clients_weights"],
        full_harvested_energy=np.array(cfg["full_harvested_energy"]),
        window_size=cfg["window_size"],
        server_deadline=cfg["server_deadline"],
        computation_times=np.array(cfg["computation_times"]),
        transmission_times=np.array(cfg["transmission_times"]),
        computation_energies=np.array(cfg["computation_energies"]),
        transmission_energies=np.array(cfg["transmission_energies"]),
        batteries_simulator=None
    )

    if system_simulator.window_size == 1:
        return MyopicStepsOptimizer(
            system_simulator=system_simulator,
            constants=cfg["constants"]
        )
    elif system_simulator.window_size == system_simulator.full_harvested_energy.shape[1]:
        return HorizonStepsOptimizer(
            system_simulator=system_simulator,
            constants=cfg["constants"]
        )
    else:
        raise NotImplementedError(
            f"Possible values for local steps optimizer are"
            f" `HorizonStepsOptimizer`, `MyopicStepsOptimize` and `BatteryStepsOptimizer`,"
            f"`window_size={system_simulator.window_size}`"
            f" and `horizon={system_simulator.full_harvested_energy.shape[1]}`"
        )


def get_local_steps_manager(
        strategy,
        n_clients,
        client_data,
        max_local_steps,
        min_local_steps,
        time_slot,
        local_steps_optimizer=None
):
    # TODO: should be updated when LocalStepsOptimizer and SystemSimulator are implemented
  
    # seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    # rng = np.random.default_rng(seed=seed)

    return LocalStepsManager(
        strategy=strategy,
        n_clients=n_clients,
        client_data=client_data,
        max_local_steps=max_local_steps,
        min_local_steps=min_local_steps,
        local_steps_optimizer=local_steps_optimizer,
        time_slot=time_slot
    )


def get_aggregator(
        aggregator_type,
        clients,
        global_learner,
        sampling_rate,
        log_freq,
        global_train_logger,
        global_test_logger,
        test_clients,
        gradients_save_path,
        metadata_save_path,
        verbose,
        classic_weights=False,
        seed=None
):
    """

    :param aggregator_type:
    :param clients:
    :param global_learner:
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients:
    :param gradients_save_path:
    :param metadata_save_path:
    :param verbose: level of verbosity
    :param classic_weights: use the weights of FedAvg to FedNova, default False
    :param seed: default is None
    :return:

    Parameters
    ----------
    classic_weights

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    if aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            gradients_save_path=gradients_save_path,
            metadata_save_path=metadata_save_path,
            verbose=verbose,
            classic_weights=classic_weights,
            seed=seed
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            gradients_save_path=gradients_save_path,
            metadata_save_path=metadata_save_path,
            verbose=verbose,
            classic_weights=classic_weights,
            seed=seed
        )
    elif aggregator_type == "fednova":
        return FedNovAggregator(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            gradients_save_path=gradients_save_path,
            metadata_save_path=metadata_save_path,
            verbose=verbose,
            classic_weights=classic_weights,
            seed=seed
        )
    else:
        raise NotImplementedError(
            "{aggregator_type} is not a possible aggregator type."
            " Available are: `no_communication`, `centralized` and `fednova`."
        )
