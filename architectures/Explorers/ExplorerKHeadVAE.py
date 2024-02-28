import torch
from architectures.Explorers.Explorer import Explorer


class ExplorerKHeadVAE(Explorer):
    def __init__(self, device):
        super().__init__(device)

        # hyperparameters
        self.min_epsilon = 0.1
        self.decay_rate = 0.9
        self.epsilon_start = 0.9

        # exploration tracking
        self.epsilon = self.epsilon_start
        self.previous_epoch = 0

        # information transfer
        self.chosen_indices = None

    @property
    def evaluation_active(self):
        return self._evaluation_active

    @evaluation_active.setter
    def evaluation_active(self, value):
        self._evaluation_active = value

    def exploration_function(self, epoch):
        """
        regulate the exploration over time
        :param epoch: current training epoch
        """
        # Decrease exploration every new epoch
        if self.previous_epoch < epoch:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            self.previous_epoch = epoch

    def get_point_from_output(self, out, epoch):
        """
        get point from encoder output
        choose specific head based on weighted probabilities
        then pass mean and variance through re-parameterization to produce single point
        """
        # init
        self.exploration_function(epoch)
        mu, log_var, weight = out

        batch_size, num_choices = weight.shape

        # choose points in batch based on probability
        random_selection_mask = torch.rand(batch_size, device=self._device) < self.epsilon

        # get max weight index for each point in batch
        argmax_indices = torch.argmax(weight, dim=1)

        # Step 3: Generate random indices for random selection
        random_indices = torch.randint(0, num_choices, (batch_size,), device=self._device)
        self.chosen_indices = torch.where(random_selection_mask, random_indices, argmax_indices)

        # get chosen mu / log_var for each point in batch
        chosen_mu = torch.gather(mu, 1, self.chosen_indices.unsqueeze(1)).squeeze(1)
        chosen_log_var = torch.gather(log_var, 1, self.chosen_indices.unsqueeze(1)).squeeze(1)

        # re-parameterize
        # compute the standard deviation
        std = torch.exp(chosen_log_var / 2)
        # compute the normal distribution with the same standard deviation
        eps = torch.randn_like(std)
        # generate a sample
        sample = chosen_mu + std * eps

        return sample
