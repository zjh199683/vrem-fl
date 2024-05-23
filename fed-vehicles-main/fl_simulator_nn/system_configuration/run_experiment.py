"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the strategy
 to select local local_steps should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

Optionally the chkpts of the model, the full gradients and the used local local_steps can be saved.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments

"""

import copy
from utils.scheduler_utils import Scheduler, import_vehicles_data
from utils.utils import *
from utils.constants import *
from utils.args import *

import numpy.linalg as LA

from torch.utils.tensorboard import SummaryWriter


def init_clients(args_, root_path, clients, apollo_records=None, lr_state_dict=None):
    """
    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_save_path: path to logs root

    :return:
        clients (List[Client]), clients_weights (np.array(n_clients))
    """
    train_iterators, val_iterators, test_iterators =\
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            root_path=root_path,
            batch_size=args_.bz,
            is_validation=args_.validation,
            clients=clients,
            apollo_records=apollo_records
        )

    clients_ = []
    clients_weights = []

    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(zip(train_iterators, val_iterators, test_iterators)):

        if train_iterator is None or test_iterator is None:
            continue

        learner =\
            get_learner(
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.local_lr,
                weight_decay=WEIGHT_DECAY,
                use_float64=args_.use_float64,
                mu=args_.mu,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                lr_state_dict=lr_state_dict
            )

        # logs_path = os.path.join(logs_save_path, "task_{}".format(task_id))
        # os.makedirs(logs_path, exist_ok=True)
        # logger = SummaryWriter(logs_path)
        logger = None

        client = get_client(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.min_local_steps,
            fit_epoch=args_.fit_epoch,
            client_id=scheduled_clients[task_id]
        )

        clients_.append(client)

        clients_weights.append(client.n_train_samples)

    clients_weights = np.array(clients_weights, dtype=float)
    clients_weights /= clients_weights.sum()

    return clients_, clients_weights


def build_experiment(args_, client_data):
    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_save_path" in args_:
        logs_save_path = args_.logs_save_path
    else:
        logs_save_path = os.path.join("logs", args_to_string(args_))

    """print("=> Clients initialization..")
    clients, clients_weights = \
        init_clients(
            args_,
            root_path=os.path.join(data_dir, "train"),
            logs_save_path=os.path.join(logs_save_path, "train"),
        )

    print("=> Test Clients initialization..")
    test_clients, _ = \
        init_clients(
            args_,
            root_path=os.path.join(data_dir, "test"),
            logs_save_path=os.path.join(logs_save_path, "test"),
        )"""

    logs_path = os.path.join(logs_save_path, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_save_path, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learner = get_learner(
        name=args_.experiment,
        device=args_.device,
        optimizer_name="sgd",
        scheduler_name=args_.lr_scheduler,
        initial_lr=args_.server_lr,
        weight_decay=0.,
        use_float64=args_.use_float64,
        mu=args_.mu,
        input_dim=args_.input_dimension,
        output_dim=args_.output_dimension,
        n_rounds=args_.n_rounds,
        seed=args_.seed
    )

    param_size = 0
    buffer_size = 0
    for param in global_learner.model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in global_learner.model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = int(param_size + buffer_size)
    local_steps_optimizer = get_local_steps_optimizer(
        cfg_file_path=args_.cfg_file_path,
        model_size=model_size,
        tx_strategy=args_.tx_strategy,
        clients_weights=None
    )

    local_steps_manager_ = get_local_steps_manager(
        strategy=args_.cpu_strategy,
        n_clients=args_.n_clients,
        client_data=client_data,
        max_local_steps=args_.max_local_steps,
        min_local_steps=args_.min_local_steps,
        time_slot=args_.time_slot,
        local_steps_optimizer=local_steps_optimizer
    )

    aggregator_ = get_aggregator(
        aggregator_type=AGGREGATOR_TYPE[args_.method],
        clients=[None] * args_.n_clients,
        global_learner=global_learner,
        sampling_rate=args_.sampling_rate,
        log_freq=args_.log_freq,
        global_train_logger=global_train_logger,
        global_test_logger=global_test_logger,
        test_clients=None,
        gradients_save_path=args_.gradients_save_path,
        metadata_save_path=args_.metadata_save_path,
        verbose=args_.verbose
    )
    
    scheduler_ = Scheduler(m=args_.n_clients)

    return aggregator_, local_steps_manager_, scheduler_, data_dir, logs_save_path, model_size


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Is CUDA available?', torch.cuda.is_available())
    args = parse_args()
    
    print('Importing vehicle data...')
    filename = '../new_corr_datasetEstim_250_2.csv'
    client_data = import_vehicles_data(filename=filename, fields=['bitrate', 'estimBitrate'],
                                       min_time=args.min_time)
    IDs = list(client_data.keys())
    
    # add batch size for each client
    batch_size = dict.fromkeys(IDs, 3)
    [client_data[client].append(batch_size[client]) for client in client_data]
    time_slot = args.time_slot
    max_latency = args.max_latency
    sim_len = args.sim_len
    aggregator, local_steps_manager, scheduler, data_dir, logs_save_path, model_size = build_experiment(args,
                                                                                                        client_data)
    scheduling = args.selection_strategy
    if args.selection_strategy == 'aoi':
        scheduling = 'optimal'
        scheduler.set_priority_params(alpha=0)

    # create optimizer for clients
    local_steps_optimizers, optSSs, condNumbs, time_start, time_end = {}, {}, {}, {}, {}
    for client in client_data:
        scheduler.add_client(client)
        time_start[client] = client_data[client][0]  # first time instant when client is available
        time_end[client] = time_start[client] + len(client_data[client][1]['bitrate']) - 1  # last time instant when client is available

    time_now = min(time_start.values())  # track current time for mobility ~ channel quality estimation
    available_clients = set()  # IDs of clients available at current time
    queued_clients = set(client_data.keys())  # IDs of clients that are still to begin

    slots = []
    steps = []
    tx_steps = []

    apollo_records = None
    root_path = os.path.join(data_dir, "train")
    if args.experiment == 'apolloscape':
        # load apollo records IDs for each client
        root_path = 'data/apolloscape/'
        with open(args.cfg_file_path, "r") as f:
            cfg = json.load(f)
        apollo_records = cfg["apollo_records"]

    print("Training..")
    pbar = tqdm(total=int((sim_len-600) / time_slot))
    lr_state_dict = None
    fraction_clients_used = []
    while time_now <= sim_len:
        # update available clients
        available_clients = set([client for client in available_clients.union(queued_clients) if
                                (time_end[client] >= time_now >= time_start[client])])
        queued_clients = queued_clients.difference(available_clients)

        if len(available_clients.union(queued_clients)) == 0:
            break
        
        elif len(available_clients) == 0:
            time_now += time_slot
            pbar.update(1)
            continue

        print("Available clients:", available_clients)
        print("Queued clients", queued_clients)
        # optimize global computation
        local_steps_manager.optimize_global(num_scheduled=args.n_clients)
        
        # optimize local computation and transmission
        local_steps_manager.optimize_local(max_latency, time_now, available_clients)
        
        # schedule clients
        if scheduling == 'all':
            scheduled_clients = list(copy.copy(available_clients))
        elif scheduling == 'random':
            scheduled_clients = scheduler.random_schedule(available_clients)
        elif scheduling == 'round_robin':
            scheduled_clients = scheduler.round_robin_schedule(available_clients)
        elif scheduling == 'optimal':
            time_feature = local_steps_manager.time_score
            if args.tx_strategy == 'min_tx_time':
                time_feature = local_steps_manager.tx_slots

            time_feature = {k: time_feature[k] for k in available_clients}
            scheduled_clients = scheduler.schedule(time_feature)
        else:
            raise NotImplementedError("Scheduling strategy requested not implemented.\n "
                                      "Choose one among:\n 'all' \n 'random' \n 'round_robin' \n 'optimal'")
                
        # compute actual latency and remove clients that exceed deadline
        latency_round = 0
        iter_clients = scheduled_clients.copy()
        print("Scheduled clients before:", scheduled_clients)
        for client in iter_clients:
            tx_bits = 0
            slots_offset_client = int(time_now - time_start[client])
            tx_slot_last_client = (slots_offset_client + local_steps_manager.comp_slots[client] +
                                   local_steps_manager.idle_slots[client])
            while tx_bits < model_size and (tx_slot_last_client - slots_offset_client) * time_slot < max_latency:
                try:
                    tx_bits += client_data[client][1]['bitrate'][tx_slot_last_client] * time_slot
                except IndexError:
                    break
                
                tx_slot_last_client += 1
                
            if tx_bits >= model_size:
                latency_client = (tx_slot_last_client - slots_offset_client) * time_slot
                if latency_round < latency_client:
                    latency_round = latency_client
                    
            else:
                scheduled_clients.remove(client)
                # if the client cannot transmit in time, 
                # the scheduler will wait till the deadline expires
                latency_round = max_latency

        print("Scheduled clients after:", scheduled_clients)
        clients, clients_weights = init_clients(args,
                                                root_path=root_path,
                                                clients=scheduled_clients,
                                                apollo_records=apollo_records,
                                                lr_state_dict=lr_state_dict)
        print("Length of learners:", len(clients))
        assert len(clients) == len(scheduled_clients)
        aggregator.set_clients(clients, clients_weights)
        local_steps = [local_steps_manager.local_steps[client] for client in scheduled_clients]
        local_steps = aggregator.set_local_steps(local_steps)

        if aggregator.save_metadata_flag or aggregator.save_gradients_flag:
            full_gradient = aggregator.get_full_gradient().cpu().numpy()

            if aggregator.save_gradients_flag:
                aggregator.save_full_gradient(full_gradient.reshape(-1, 1))

            if aggregator.save_metadata_flag:
                metadata = {
                    "client_weights": aggregator.clients_weights.cpu().numpy().tolist(),
                    "local_steps": local_steps.tolist(),
                    "server_lr": server_lr,
                    "gradient_norm": float(LA.norm(full_gradient))
                }

                aggregator.save_metadata(metadata)

        fraction_clients_used.append(len(scheduled_clients) / args.n_clients)
        #aggregator.mix(local_steps_manager)
        #lr_state_dict = clients[0].learner.lr_scheduler.state_dict()
        #print("State dict:", lr_state_dict)
        time_now += np.ceil(latency_round / time_slot)
        pbar.update(int(np.ceil(latency_round / time_slot)))

    if aggregator.full_gradients_file is not None:
        aggregator.full_gradients_file.close()

    if "chkpts_save_path" in args:
        os.makedirs(args.chkpts_save_path, exist_ok=True)
        aggregator.save_state(args.chkpts_save_path)

    with open('frac_clients_used_{}_{}.pkl'.format(args.selection_strategy, args.n_clients), 'wb') as f:
        pickle.dump(fraction_clients_used, f)
