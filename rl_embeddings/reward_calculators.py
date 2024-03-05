import abc
import torch
import torch.nn as nn
import torch.nn.functional as f


class RewardCalculator(nn.Module, abc.ABC):
    def __init__(self, device):
        super().__init__()
        self._device = device

        # general trivial reward
        self._trivial_reward = torch.tensor([0.], requires_grad=True)

    @abc.abstractmethod
    def forward(self, s_o, enc_o, exp_o, p_o, tr_o, dec_o,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        compute reward for encoder and decoder
        :param s_o: direct output of Sampler object
        :param enc_o: direct output of encoder agent
        :param exp_o: direct output of Explorer object
        :param p_o: direct output of PropertyCalculator object
        :param tr_o: direct output of Transmitter object
        :param dec_o: direct output of decoder agent
        must return three torch tensors of shape [1] with gradients activated
            - encoder reward: tensor for training the encoder
            - decoder reward: tensor for training the decoder
            - total reward : tensor for jointly training encoder and decoder
        """
        raise NotImplementedError


class RewardCalculatorVAE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)

        # hyperparameters
        self.success_weight = 1
        self.kl_weight = 1

    def forward(self, s_o, enc_o, exp_o, p_o, tr_o, dec_o) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        compute reward for encoder and decoder
        a VAE only trains the encoder and decoder jointly
        therefore only the total_reward is non-trivial
        """
        # get information from different steps of embedding process
        mu, log_var = enc_o
        x_a, _ = s_o
        x_b = dec_o

        # KL term with prior as gaussian
        kl_divergence = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())

        # reconstruction term
        total_loss = f.mse_loss(x_b, x_a, reduction='sum') * self.success_weight + kl_divergence * self.kl_weight
        total_reward = (-1) * total_loss

        return self._trivial_reward, self._trivial_reward, total_reward


class RewardCalculatorKHeadVAE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)

        # hyperparameters
        self.success_weight = 1

    def forward(self, s_o, enc_o, exp_o, p_o, tr_o, dec_o,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        encoder and decoder are trained jointly
        reward also considers the weight of the chosen head
        """
        # get information from different steps of embedding process
        x_a, _ = s_o
        x_b = dec_o
        mu, log_var, weight = enc_o
        _, chosen_indices, chosen_mu, chosen_log_var = exp_o

        # tensor for multiplying reward with probability
        mean_weights = weight.gather(1, chosen_indices.unsqueeze(1))

        # similar to VAE loss, without KL term
        variance = torch.exp(chosen_log_var)
        surprise = variance + torch.square(chosen_mu)
        success = f.mse_loss(x_a, x_b)
        total_loss = torch.sum((surprise + (success * self.success_weight)) * mean_weights)
        total_reward = (-1) * total_loss

        return self._trivial_reward, self._trivial_reward, total_reward


class RewardCalculatorUMAP(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)

    def compute_encoder_reward(self, s_o, enc_o, exp_o, p_o, tr_o, dec_o):
        # get information from different steps of embedding process
        low_dim_prop, high_dim_prop = p_o
        p1, _, _, _ = s_o
        x_a, _ = p1
        x_b = dec_o

        # encoder reward based on properties
        encoder_loss = f.binary_cross_entropy(low_dim_prop, high_dim_prop)
        encoder_reward = (-1) * encoder_loss
        return encoder_reward

    def compute_decoder_reward(self, s_o, enc_o, exp_o, p_o, tr_o, dec_o):
        # get information from different steps of embedding process
        low_dim_prop, high_dim_prop = p_o
        p1, _, _, _ = s_o
        x_a, _ = p1
        x_b = dec_o

        # decoder reward based on reconstruction
        decoder_loss = f.mse_loss(x_b, x_a, reduction='sum')
        decoder_reward = (-1) * decoder_loss
        return decoder_reward

    def forward(self, s_o, enc_o, exp_o, p_o, tr_o, dec_o) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        encoder is trained on high- / low-dim property differences
        decoder is trained on reconstruction
        """
        # get information from different steps of embedding process
        low_dim_prop, high_dim_prop = p_o
        p1, _, _, _ = s_o
        x_a, _ = p1
        x_b = dec_o

        # encoder reward based on properties
        encoder_loss = f.binary_cross_entropy(low_dim_prop, high_dim_prop)
        encoder_reward = (-1) * encoder_loss

        # decoder reward based on reconstruction
        decoder_loss = f.mse_loss(x_b, x_a, reduction='sum')
        decoder_reward = (-1) * decoder_loss

        return encoder_reward, decoder_reward, self._trivial_reward


class RewardCalculatorVarianceVAE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)

    def forward(self, s_o, enc_o, exp_o, p_o, tr_o, dec_o,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        train encoder and decoder jointly on only the reconstruction
        """
        # get information from different steps of embedding process
        x_a, _ = s_o
        x_b = dec_o

        # reconstruction term
        total_loss = f.mse_loss(x_b, x_a, reduction='sum')
        total_reward = (-1) * total_loss

        return self._trivial_reward, self._trivial_reward, total_reward
