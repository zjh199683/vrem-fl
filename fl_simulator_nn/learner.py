import torch
from copy import deepcopy


class Learner:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner

    criterion (torch.nn.modules.loss): loss function used to train the `model`, should have reduction="none"

    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar

    device (str or torch.device):

    optimizer (torch.optim.Optimizer):

    lr_scheduler (torch.optim.lr_scheduler):

    is_binary_classification (bool): whether to cast labels to float or not, if `BCELoss`
    is used as criterion this should be set to True

    use_float64 (bool): if True a 64-bits representation is used to store the model

    Methods
    ------
    compute_gradients_and_loss:

    optimizer_step: perform one optimizer step, requires the gradients to be already computed.

    fit_batch: perform an optimizer step over one batch

    fit_epoch:

    fit_batches: perform successive optimizer local_steps over successive batches

    fit_epochs:

    evaluate_iterator: evaluate `model` on an iterator

    gather_losses:

    get_param_tensor: get `model` parameters as a unique flattened tensor

    free_memory: free the memory allocated by the model weights

    free_gradients:
    """

    def __init__(
            self, 
            model,
            criterion,
            metric,
            device,
            optimizer,
            lr_scheduler=None,
            is_binary_classification=False,
            use_float64=False
    ):

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_binary_classification = is_binary_classification

        self.use_float64 = use_float64

        self.model_dim = int(self.get_param_tensor().shape[0])

        if self.use_float64:
            self.model = self.model.to(torch.double)

    def compute_gradients_and_loss(self, batch, weights=None, accumulate_gradients=False):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the weights of each sample or None
        :type weights: torch.tensor or None
        :param accumulate_gradients: if `False` the gradient is set to zero before computing the gradient;
         default is `False`
        :type accumulate_gradients
        :return:
            loss (float)

        """
        self.model.train()

        x, y, indices = batch

        x = x.to(self.device).type(torch.float32)
        if self.use_float64:
            x = x.type(torch.float64)

        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)
            if self.use_float64:
                y = y.type(torch.float64)

        if not accumulate_gradients:
            self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        return loss.detach()

    def compute_full_gradient(self, iterator):
        """
        compute full gradient on all samples of an iterator
        :param iterator:
        :return:
            None

        """
        self.optimizer.zero_grad()
        for batch in iterator:
            self.compute_gradients_and_loss(batch, accumulate_gradients=True)

    def optimizer_step(self):
        """
         perform one optimizer step, requires the gradients to be already computed.

        """
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def fit_batch(self, batch, weights=None, norm=True):
        """
        perform an optimizer step over one batch drawn from `iterator`

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.item()
            metric.item()

        """
        self.model.train()

        x, y, indices = batch

        x = x.to(self.device).type(torch.float32)
        if self.use_float64:
            x = x.type(torch.float64)

        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)
            if self.use_float64:
                y = y.type(torch.float64)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        if isinstance(y_pred, dict):
            y_pred = y_pred['out']
        loss_vec = self.criterion(y_pred, y)
        # TODO: check division when using jaccard
        #metric = self.metric(y_pred, y)  # / len(y)
        self.metric.add_sample(y_pred.detach().argmax(dim=1), y)
        metric = self.metric.percent_mIoU()
        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()
        
        grad_norm = None
        if norm:
            # grad_norm = 0
            # for param in self.model.parameters():
            #     param_grad_norm = param.grad.data.norm(2)
            #     grad_norm += param_grad_norm.item() ** 2
            #
            # grad_norm = grad_norm ** .5
            grads = [
                param.grad.detach().flatten()
                for param in self.model.parameters()
                if param.grad is not None
            ]
            grad_norm = torch.cat(grads).norm()

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.item(), metric.item(), grad_norm

    def fit_epoch(self, iterator, weights=None):
        """
        perform several optimizer local_steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.item()
            metric.item()

        """
        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            if self.use_float64:
                x = x.type(torch.float64)

            y = y.to(self.device)
            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)
                if self.use_float64:
                    y = y.type(torch.float64)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y)
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()
            loss.backward()

            self.optimizer.step()

            global_loss += loss.item() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).item()

        return global_loss / n_samples, global_metric / n_samples * y.size(0)

    def fit_epochs(self, iterator, n_epochs, weights=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch(iterator, weights)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, _ in iterator:
            x = x.to(self.device).type(torch.float32)
            if self.use_float64:
                x = x.type(torch.float64)

            y = y.to(self.device)
            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)
                if self.use_float64:
                    y = y.type(torch.float64)

            with torch.no_grad():
                y_pred = self.model(x)
                if isinstance(y_pred, dict):
                    y_pred = y_pred['out']
                global_loss += self.criterion(y_pred, y).sum().item()
                global_metric += self.metric(y_pred, y).item()

            n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples * y.size(0)

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor

        :return: torch.tensor

        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def set_param_tensor(self, param_tensor):
        """
        sets the parameters of the model from `param_tensor`

        :param param_tensor: torch.tensor of shape (`self.model_dim`,)

        """
        param_tensor = param_tensor.to(self.device)

        current_index = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            current_dimension = param.data.view(-1, ).shape[0]

            param.data = \
                param_tensor[current_index: current_index + current_dimension].reshape(param_shape)

            current_index += current_dimension

    def set_grad_tensor(self, grad_tensor):
        """

        :param grad_tensor: torch.tensor of shape (`self.model_dim`,)

        """
        grad_tensor = grad_tensor.to(self.device)

        current_index = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            current_dimension = param.data.view(-1, ).shape[0]

            if param.grad is None:
                param.grad = \
                    grad_tensor[current_index: current_index + current_dimension].reshape(param_shape)
            else:
                param.grad.data = \
                    grad_tensor[current_index: current_index + current_dimension].reshape(param_shape)

            current_index += current_dimension

    def free_memory(self):
        """
        free the memory allocated by the model weights
        
        """
        del self.optimizer
        del self.model

    def free_gradients(self):
        """
        free memory allocated by gradients

        """
        self.optimizer.zero_grad(set_to_none=True)


class LanguageModelingLearner(Learner):
    def fit_epoch(self, iterator, weights=None):

        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device)
            y = y.to(self.device)

            n_samples += y.size(0)

            chunk_len = y.size(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)
            loss_vec = self.criterion(y_pred, y)

            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
            else:
                loss = loss_vec.mean()

            loss.backward()

            self.optimizer.step()

            global_loss += loss.item() * loss_vec.size(0) / chunk_len
            global_metric += self.metric(y_pred, y).item() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def compute_gradients_and_loss(self, batch, weights=None, accumulate_gradients=False):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :param accumulate_gradients: if `False` the gradient is set to zero before computing the gradient;
                                     default is `False`
        :type accumulate_gradients
        :return:
            loss

        """
        # TODO
        raise NotImplementedError

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y, _ in iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                n_samples += y.size(0)

                chunk_len = y.size(1)

                y_pred = self.model(x)
                global_loss += self.criterion(y_pred, y).sum().item() / chunk_len
                global_metric += self.metric(y_pred, y).item() / chunk_len

        return global_loss / n_samples, global_metric / n_samples
