import torch
import architectures.rl_vae as rl_vae


class ConstantExplorationRLVAE(rl_vae.RlVae):
    def __init__(self, device, latent_dimensions=2):
        super().__init__(device, latent_dimensions)
        self.arch_name = "ConstantExplorationRL-VAE"
        self.encoder_agent = rl_vae.MeanEncoderAgent(latent_dimensions).to(self.device)
        self.exploration_rate = 1
        self.exploration_function = self.constant_exploration_function
        self.reward_function = self.non_exploration_reward_function

    def constant_exploration_function(self):
        """
        set the exploration rate to a constant value across all dimensions
        """
        logvar = torch.tensor([self.exploration_rate] * self.latent_dimensions).to(self.device)
        return logvar
