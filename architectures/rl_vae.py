import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils
import torch.distributions
import matplotlib.pyplot as plt

# module init
torch.manual_seed(0)


class EncoderAgent(nn.Module):
    def __init__(self, latent_dims):
        super(EncoderAgent, self).__init__()
        self.linear1 = nn.Linear(3, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linearM = nn.Linear(1024, latent_dims)
        self.linearS = nn.Linear(1024, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = functional.relu(self.linear1(x))
        x = functional.relu(self.linear2(x))
        # mean
        mu = self.linearM(x)
        # variance
        log_var = self.linearS(x)
        return mu, log_var


class DecoderAgent(nn.Module):
    def __init__(self, latent_dims):
        super(DecoderAgent, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 3)

    def forward(self, z):
        z = functional.relu(self.linear1(z))
        z = functional.relu(self.linear2(z))
        z = self.linear3(z)
        return z


class RlVae:
    def __init__(self, device, latent_dimensions=2):
        self.device = device
        self.encoder_agent = EncoderAgent(latent_dimensions).to(device)
        self.decoder_agent = DecoderAgent(latent_dimensions).to(device)
        self.optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()) + list(self.decoder_agent.parameters()))

        self.latent_dimensions = 2
        self.verbose = True
        self.arch_name = "RL-VAE"

        self.transmit_function = self.transmit_identity
        self.reward_function = self.standard_reward_function

        self.avg_loss_li = []
        self.total_loss_li = []

    def console_log(self, message):
        if not self.verbose:
            return
        print(message)

    @staticmethod
    def transmit_identity(z_a):
        return z_a

    @staticmethod
    def standard_reward_function(x_a, x_b, mean, logvar, z_a):
        """
        the RL-VAE reward function including:
        1. the success term -> reconstruction
        2. the KL-divergence as the exploration and surprise
        """
        success = functional.mse_loss(x_a, x_b, reduction='sum')
        variance = torch.exp(logvar)
        exploration = -0.5 * torch.sum(
            torch.log(2 * torch.pi * variance) + ((z_a - mean) / torch.sqrt(variance)).pow(2))
        surprise = -0.5 * torch.sum(math.log(2 * torch.pi) + z_a.pow(2))
        return -(success + (exploration - surprise))

    @staticmethod
    def non_exploration_reward_function(x_a, x_b, mean, logvar, z_a):
        """
        the RL-VAE reward function without the exploration term
        """
        success = functional.mse_loss(x_a, x_b, reduction='sum')
        surprise = -0.5 * torch.sum(math.log(2 * torch.pi) + z_a.pow(2))
        return -(success - surprise)

    @staticmethod
    def re_parameterize(mu, log_var):
        """
        sample from a gaussian distribution with mean: mu and variance: log_var
        this samples using the re_parameterization trick so that the gradients can be calculated
        :param mu:
        :param log_var:
        :return:
        """
        # compute the standard deviation
        std = torch.exp(log_var / 2)
        # compute the normal distribution with the same standard deviation
        eps = torch.randn_like(std)
        # generate a sample
        sample = mu + std * eps
        return sample

    def plot_latent(self, data_loader, save_as, num_batches=100):
        """
        plot the generated latent space using matplotlib
        :param data_loader: the dataloader converted from toy data
        :param save_as: name and path of file to save image in
        :param num_batches: max number of batches
        :return:
        """
        self.encoder_agent.to('cpu')
        for i, (x, y) in enumerate(data_loader):
            mu, log_var = self.encoder_agent(x.to('cpu'))
            z = self.re_parameterize(mu, log_var)
            z = z.to('cpu').detach().numpy()
            # Assume that `y` contains the colors and is in the shape (batch_size, 4)
            # We take only the first three values assuming they correspond to RGB colors
            colors = y[:, :3].to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=colors)
            if i >= num_batches:
                plt.colorbar()
                break
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'{self.arch_name} projection')
        plt.savefig(save_as)
        plt.close()
        self.console_log("Finished plotting latent")

    def plot_loss(self, save_as):
        """
        plot the progression of the loss
        shows the average loss per epoch
        :param save_as: path and name of file to save figure
        :return:
        """
        # Plotting the average loss per epoch
        plt.plot(range(len(self.avg_loss_li)), self.avg_loss_li)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Average Loss per Epoch')
        plt.savefig(save_as)
        plt.close()
        self.console_log("Finished plotting loss")

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

            for x, y in training_data_loader:
                # get data
                x_a = x.to(self.device)

                # encode data point (encoder policy action)
                mean, logvar = self.encoder_agent(x_a)

                # re-parameterize to get a sample from the approximate posterior
                z_a = self.re_parameterize(mean, logvar)

                # transmit through noisy channel
                z_b = self.transmit_function(z_a)

                # decode (decoder policy action)
                x_b = self.decoder_agent(z_b)

                # compute reward / loss
                reward = self.reward_function(x_a, x_b, mean, logvar, z_a)
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
