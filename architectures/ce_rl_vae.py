import random
import torch
import torch.nn.functional as functional
import architectures.rl_vae as rl_vae


class ConstantExplorationRLVAE(rl_vae.RlVae):
    def __init__(self, device, input_dim, latent_dimensions=2):
        super().__init__(device, input_dim, latent_dimensions)
        self.arch_name = "ConstantExplorationRL-VAE"
        self.encoder_agent = rl_vae.MeanEncoderAgent(input_dim, latent_dimensions).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.encoder_agent.parameters()) + list(self.decoder_agent.parameters()),
            weight_decay=1e-2
        )
        self.epsilon = 0.5

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
