class Client(object):
    r"""Implements one clients

    Attributes
    ----------
    learner
    train_iterator
    val_iterator
    test_iterator
    train_loader
    n_train_samples
    n_test_samples
    local_steps
    fit_epoch
    logger

    Methods
    ----------
    __init__
    step
    write_logs
    get_full_gradient

    """
    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            fit_epoch,
            client_id=None
    ):

        self.learner = learner
        self.device = self.learner.device
        self.model_dim = self.learner.model_dim
        self.client_id = client_id

        self.binary_classification_flag = self.learner.is_binary_classification

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.train_loader = iter(self.train_iterator)

        self.default_local_steps_init = 5
        self.local_steps = local_steps
        self.fit_epoch = fit_epoch

        self.counter = 0
        self.logger = logger
        
    @property
    def local_steps(self):
        return self.__local_steps

    @local_steps.setter
    def local_steps(self, local_steps):
        if local_steps <= 0:
            local_steps = 1

        self.__local_steps = int(local_steps)

    def get_lr(self):
        for param_groups in self.learner.optimizer.param_groups:
            return param_groups['lr']

    def get_next_batch(self):
        try:
            batch = next(self.train_loader)
        except StopIteration:
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)

        return batch

    def step(self, local_steps_manager):
        self.counter += 1
        import time
        s = time.time()
        if self.fit_epoch:
            self.learner.fit_epochs(iterator=self.train_iterator, n_epochs=self.local_steps)

        else:
            loss = []; grad_norm = []
            for _ in range(self.default_local_steps_init):
                batch = self.get_next_batch()
                loss_batch, _, grad_norm_batch = self.learner.fit_batch(batch=batch, norm=True)
                loss.append(loss_batch)
                grad_norm.append(grad_norm_batch)
                
            avg_loss = sum(loss) / len(loss)
            grad_norm = sum(grad_norm) / len(grad_norm)
            self.local_steps = local_steps_manager.adjust_local_steps(self.client_id, avg_loss, grad_norm)
            print(self.local_steps)
            for _ in range(self.local_steps - self.default_local_steps_init):
                batch = self.get_next_batch()
                loss_batch, _, _ = self.learner.fit_batch(batch=batch)
                loss.append(loss_batch)
            self.train_loss = sum(loss) / len(loss)
            print(time.time()-s)


    def get_full_gradient(self):
        """
        compute full gradient on all dataset

        :return:
            torch.tensor((self.model_dim, ), device=self.device)

        """
        self.learner.compute_full_gradient(self.val_iterator)
        return self.learner.get_grad_tensor()

    def write_logs(self):

        train_loss, train_acc = self.learner.evaluate_iterator(self.val_iterator)
        test_loss, test_acc = self.learner.evaluate_iterator(self.test_iterator)

        # self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        # self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        # self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        # self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc
    