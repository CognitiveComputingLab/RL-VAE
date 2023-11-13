import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils
import torch.distributions
import matplotlib.pyplot as plt

# module init
torch.manual_seed(0)


class GeneralModel(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(GeneralModel, self).__init__()
        self.layers = nn.ModuleList()
        for h_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, h_dim),
                nn.ReLU(),
            ))
            input_dim = h_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderAgent(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(EncoderAgent, self).__init__()
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048])
        self.linearM = nn.Linear(2048, latent_dims)
        self.linearS = nn.Linear(2048, latent_dims)

    def forward(self, x):
        x = self.gm(x)
        mu = self.linearM(x)
        log_var = self.linearS(x)
        return mu, log_var


class MeanEncoderAgent(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(MeanEncoderAgent, self).__init__()
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048])
        self.linearM = nn.Linear(2048, latent_dims)

    def forward(self, x):
        x = self.gm(x)
        mu = self.linearM(x)
        return mu


class DecoderAgent(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(DecoderAgent, self).__init__()
        self.gm = GeneralModel(latent_dims, [512, 1024, 2024])
        self.linear = nn.Linear(2024, input_dim)

    def forward(self, z):
        z = self.gm(z)
        z = self.linear(z)
        return z


class RlVae:
    def __init__(self, device, input_dim, latent_dimensions=2):
        self.device = device
        self.encoder_agent = EncoderAgent(input_dim, latent_dimensions).to(device)
        self.decoder_agent = DecoderAgent(input_dim, latent_dimensions).to(device)
        self.optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()) + list(self.decoder_agent.parameters()))

        self.latent_dimensions = latent_dimensions
        self.input_dim = input_dim

        self.verbose = True
        self.arch_name = "RL-VAE"

        self.transmit_function = self.transmit_identity
        self.reward_function = self.standard_reward_function
        self.exploration_function = lambda: 1
        self.success_weight = 1

        self.avg_loss_li = []
        self.total_loss_li = []

    def console_log(self, message):
        if not self.verbose:
            return
        print(message)

    @staticmethod
    def transmit_identity(z_a):
        return z_a

    def standard_reward_function(self, x_a, x_b, mean, logvar):
        """
        the RL-VAE reward function
        """
        variance = torch.exp(logvar)
        exploration = logvar
        surprise = -variance - torch.square(mean)
        success = -functional.mse_loss(x_a, x_b, reduction='sum')
        result = torch.sum(exploration + surprise) + success * self.success_weight
        return result

    def non_exploration_reward_function(self, x_a, x_b, mean, logvar):
        """
        the RL-VAE reward function without the exploration term
        """
        variance = torch.exp(logvar)
        surprise = -variance - torch.square(mean)
        success = -functional.mse_loss(x_a, x_b, reduction='sum')
        result = torch.sum(surprise) + success * self.success_weight
        return result

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
            # compute encoded datapoint
            result = self.encoder_agent(x.to('cpu'))
            if type(result) is tuple:
                mean, logvar = result
            else:
                mean = result
                logvar = self.exploration_function()
            mean = mean.to('cpu')
            logvar = logvar.to('cpu')
            z = self.re_parameterize(mean, logvar)
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
                result = self.encoder_agent(x_a)
                if type(result) is tuple:
                    mean, logvar = result
                else:
                    mean = result
                    logvar = self.exploration_function()

                # re-parameterize to get a sample from the approximate posterior
                z_a = self.re_parameterize(mean, logvar)

                # transmit through noisy channel
                z_b = self.transmit_function(z_a)

                # decode (decoder policy action)
                x_b = self.decoder_agent(z_b)

                # compute reward / loss
                reward = self.reward_function(x_a, x_b, mean, logvar)
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

    def save_model(self, path):
        """
        save the encoder and decoder separately
        """
        torch.save(self.encoder_agent.state_dict(), f"{path}/{self.arch_name}-encoder.png")
        torch.save(self.decoder_agent.state_dict(), f"{path}/{self.arch_name}-decoder.png")
