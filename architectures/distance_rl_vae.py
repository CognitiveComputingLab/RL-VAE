import random
import numpy as np
import torch
import torch.nn.functional as functional
from tqdm import tqdm
import architectures.rl_vae as rl_vae
from helper import symmetric_kl_divergence_2d, scale_to_01


class DistanceRLVAE(rl_vae.RlVae):
    def __init__(self, device, input_dim, num_heads, latent_dimensions=2):
        super().__init__(device, input_dim, latent_dimensions)
        self.arch_name = "DistanceRL-VAE"
        self.encoder_agent = rl_vae.KMeanEncoderAgent(input_dim, latent_dimensions, num_heads).to(device)
        self.decoder_agent = rl_vae.DecoderAgent(input_dim, latent_dimensions).to(device)
        self.optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()) + list(self.decoder_agent.parameters()))

        self.distance_reward_weight = 1
        self.max_distribution_distance = 1000

    def extended_reward_function(self, x_a, x_b, mean, logvar, mean_weights, all_mus, all_logvar):
        """
        the RL-VAE reward function
        """
        # normal RL-VAE reward
        variance = torch.exp(logvar)
        surprise = variance + torch.square(mean)
        success = functional.mse_loss(x_a, x_b)
        result = -torch.sum((surprise + (success * self.success_weight)) * mean_weights)

        # additional reward for the average distance between distributions
        average_distance = symmetric_kl_divergence_2d(all_mus, all_logvar)
        average_distance = min(average_distance, self.max_distribution_distance)
        scaled_distance = scale_to_01(average_distance, 0, self.max_distribution_distance)
        scaled_distance = (scaled_distance * self.distance_reward_weight) - self.distance_reward_weight
        result += scaled_distance
        return result

    def train(self, training_data_loader, epochs=100):
        """
        train the RL-VAE model
        :return:
        """
        self.console_log(f"Starting training for: {self.arch_name}")

        # training loop
        for epoch in range(epochs):
            self.console_log(f"---------------------- EPOCH {epoch} ----------------------")

            # keep track of loss
            total_loss = 0
            counter = 0

            for x, _ in tqdm(training_data_loader, disable=False):
                x_a = x.to(self.device)

                # run through neural network
                mus, logvar, weights = self.encoder_agent(x_a)

                # handle exploration
                chosen_mus, chosen_log_vars, chosen_indices = self.exploration_function(mus, logvar, weights, epoch)

                # sample / re-parameterize
                z_a = self.re_parameterize(chosen_mus, chosen_log_vars)

                # transmit through noisy channel
                z_b = self.transmit_function(z_a)

                # decode (decoder policy action)
                x_b = self.decoder_agent(z_b)

                # compute reward / loss
                reward = self.extended_reward_function(x_a, x_b, chosen_mus, chosen_log_vars,
                                                       weights.gather(1, chosen_indices.unsqueeze(1)), mus, logvar)
                loss = -reward
                total_loss += float(loss)
                counter += 1

                # gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # print epoch loss
            avg_loss = total_loss / counter
            self.total_loss_li.append(total_loss)
            self.avg_loss_li.append(avg_loss)
            self.console_log(f"total loss: {total_loss}")
            self.console_log(f"average loss: {avg_loss}")
            if epoch % 10 == 0:
                self.plot_latent(training_data_loader, f"images/{self.arch_name}-epoch-{epoch}-latent.png")