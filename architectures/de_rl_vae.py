import random
import numpy as np
import torch
import torch.nn.functional as functional
import architectures.rl_vae as rl_vae


class DecreasingExplorationRLVAE(rl_vae.RlVae):
    def __init__(self, device, input_dim, num_heads, latent_dimensions=2):
        super().__init__(device, input_dim, latent_dimensions)
        self.arch_name = "DecreasingExplorationRL-VAE"
        self.encoder_agent = rl_vae.KMeanEncoderAgent(input_dim, latent_dimensions, num_heads).to(device)
        self.decoder_agent = rl_vae.DecoderAgent(input_dim, latent_dimensions).to(device)
        self.optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()) + list(self.decoder_agent.parameters()))

        self.epsilon = 1
        self.decay_rate = 0.999
        self.min_epsilon = 0.001
        self.previous_epoch = float("-inf")

    def reward_function(self, x_a, x_b, mean, logvar, mean_weights):
        """
        the RL-VAE reward function
        """
        variance = torch.exp(logvar)
        surprise = variance + torch.square(mean)
        success = functional.mse_loss(x_a, x_b)
        result = -torch.sum((surprise + (success * self.success_weight)) * mean_weights)
        return result

    def exploration_function(self, mus, logvar, weights, epoch):
        """
        set the exploration rate to a constant value across all dimensions
        """
        # decrease exploration every new epoch
        if self.previous_epoch < epoch:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            self.previous_epoch = epoch

        chosen_mus = []
        chosen_log_vars = []
        chosen_indices = []

        for i in range(weights.shape[0]):
            if random.random() < self.epsilon:
                # Randomly select a mean
                random_index = random.randint(0, weights.shape[1] - 1)
                chosen_mu = mus[i, random_index]
                chosen_log_var = logvar[i, random_index]
                chosen_indices.append(random_index)
            else:
                # Use argmax to select a mean
                argmax_index = torch.argmax(weights[i])
                chosen_mu = mus[i, argmax_index]
                chosen_log_var = logvar[i, argmax_index]
                chosen_indices.append(argmax_index)

            chosen_mus.append(chosen_mu)
            chosen_log_vars.append(chosen_log_var)

        chosen_mus = torch.stack(chosen_mus)
        chosen_log_vars = torch.stack(chosen_log_vars)
        chosen_indices = torch.tensor(chosen_indices).to(self.device)
        return chosen_mus, chosen_log_vars, chosen_indices


class SoftmaxDecreasingExplorationRLVAE(rl_vae.RlVae):
    def __init__(self, device, input_dim, num_heads, latent_dimensions=2):
        super().__init__(device, input_dim, latent_dimensions)
        self.arch_name = "SoftmaxDecreasingExplorationRL-VAE"
        self.encoder_agent = rl_vae.KMeanEncoderAgent(input_dim, latent_dimensions, num_heads).to(device)
        self.decoder_agent = rl_vae.DecoderAgent(input_dim, latent_dimensions).to(device)
        self.optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()) + list(self.decoder_agent.parameters()))

        self.epsilon = 1
        self.decay_rate = 0.999
        self.min_epsilon = 0.001
        self.previous_epoch = float("-inf")

    def reward_function(self, x_a, x_b, mean, logvar, mean_weights):
        """
        the RL-VAE reward function
        """
        variance = torch.exp(logvar)
        surprise = variance + torch.square(mean)
        success = functional.mse_loss(x_a, x_b)
        result = -torch.sum((surprise + (success * self.success_weight)) * mean_weights)
        return result

    def exploration_function(self, mus, logvar, weights, epoch):
        """
        Set the exploration rate to a constant value across all dimensions and
        select the mean randomly using the weights as probabilities.
        """
        # Decrease exploration every new epoch
        if self.previous_epoch < epoch:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            self.previous_epoch = epoch

        chosen_mus = []
        chosen_log_vars = []
        chosen_indices = []

        for i in range(weights.shape[0]):
            if random.random() < self.epsilon:
                # Apply softmax to the weights
                softmax_weights = torch.softmax(weights[i], dim=0)

                # Use softmax probabilities to randomly select a mean
                random_index = torch.multinomial(softmax_weights, 1).item()
                chosen_mu = mus[i, random_index]
                chosen_log_var = logvar[i, random_index]
                chosen_indices.append(random_index)
            else:
                # Use argmax to select a mean
                argmax_index = torch.argmax(weights[i])
                chosen_mu = mus[i, argmax_index]
                chosen_log_var = logvar[i, argmax_index]
                chosen_indices.append(argmax_index)

            chosen_mus.append(chosen_mu)
            chosen_log_vars.append(chosen_log_var)

        chosen_mus = torch.stack(chosen_mus)
        chosen_log_vars = torch.stack(chosen_log_vars)
        chosen_indices = torch.tensor(chosen_indices).to(self.device)
        return chosen_mus, chosen_log_vars, chosen_indices


