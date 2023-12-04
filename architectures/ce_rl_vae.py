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
        self.initial_exploration = 1

    def reward_function(self, x_a, x_b, mean, logvar):
        """
        the RL-VAE reward function without the exploration term
        """
        variance = torch.exp(logvar)
        surprise = -variance - torch.square(mean)
        success = -functional.mse_loss(x_a, x_b)
        result = torch.sum(surprise + success * self.success_weight)
        return result

    def exploration_function(self, epoch):
        """
        set the exploration rate to a constant value across all dimensions
        """
        logvar = torch.tensor([self.initial_exploration] * self.latent_dimensions).to(self.device)
        return logvar

    def train(self, training_data_loader, epochs=100):
        super().train(training_data_loader, epochs)
