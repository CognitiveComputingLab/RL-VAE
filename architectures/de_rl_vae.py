import numpy as np
import torch
import torch.nn.functional as functional
import architectures.rl_vae as rl_vae


class DecreasingExplorationRLVAE(rl_vae.RlVae):
    def __init__(self, device, input_dim, latent_dimensions=2):
        super().__init__(device, input_dim, latent_dimensions)
        self.arch_name = "DecreasingExplorationRL-VAE"
        self.encoder_agent = rl_vae.MeanEncoderAgent(input_dim, latent_dimensions).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()) + list(self.decoder_agent.parameters()))

        self.initial_exploration = 2
        self.exploration_decay = 0.99
        self.previous_epoch = 0
        self.min_exploration = 0.1
        self.current_exploration = self.initial_exploration

    def reward_function(self, x_a, x_b, mean, logvar):
        """
        the RL-VAE reward function without the exploration term
        """
        variance = torch.exp(logvar)
        surprise = -variance - torch.square(mean)
        success = -functional.mse_loss(x_a, x_b, reduction='sum')
        result = torch.sum(surprise) + success * self.success_weight
        return result

    def exploration_function(self, epoch):
        """
        set the exploration rate to a constant value across all dimensions
        """
        if epoch > self.previous_epoch:
            self.previous_epoch = epoch
            self.current_exploration *= self.exploration_decay

        logvar = torch.tensor([self.current_exploration] * self.latent_dimensions).to(self.device)
        self.current_exploration = max(self.min_exploration, self.current_exploration)
        return logvar


