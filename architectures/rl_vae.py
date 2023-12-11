import datetime
import random
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
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linearM = nn.Linear(4096, latent_dims)
        self.linearS = nn.Linear(4096, latent_dims)

    def forward(self, x):
        x = self.gm(x)
        mu = self.linearM(x)
        log_var = self.linearS(x)
        return mu, log_var


class MeanEncoderAgent(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(MeanEncoderAgent, self).__init__()
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linearM = nn.Linear(4096, latent_dims)

    def forward(self, x):
        x = self.gm(x)
        mu = self.linearM(x)
        return mu


class KMeanEncoderAgent(nn.Module):
    def __init__(self, input_dim, latent_dims, k):
        super(KMeanEncoderAgent, self).__init__()
        self.k = k
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linearM = nn.Linear(4096, k * latent_dims)
        self.linearS = nn.Linear(4096, k * latent_dims)
        self.weight_gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linear_weight = nn.Linear(4096, k)

    def forward(self, x):
        x2 = self.gm(x)

        # output the means
        mu = self.linearM(x2)
        batch_size = mu.size(0)
        mu = mu.view(batch_size, self.k, -1)

        # output the log-vars
        logvar = self.linearS(x2)
        batch_size = logvar.size(0)
        logvar = logvar.view(batch_size, self.k, -1)

        weights = self.weight_gm(x)
        weights = self.linear_weight(weights)
        weights = nn.functional.softmax(weights, dim=1)

        return mu, logvar, weights


class DecoderAgent(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(DecoderAgent, self).__init__()
        self.gm = GeneralModel(latent_dims, [512, 1024, 2024, 4096])
        self.linear = nn.Linear(4096, input_dim)

    def forward(self, z):
        z = self.gm(z)
        z = self.linear(z)
        return z


class RlVae:
    def __init__(self, device, input_dim, num_heads, latent_dimensions=2):
        self.device = device
        self.encoder_agent = KMeanEncoderAgent(input_dim, latent_dimensions, num_heads).to(device)
        self.decoder_agent = DecoderAgent(input_dim, latent_dimensions).to(device)
        self.optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()) + list(self.decoder_agent.parameters()))

        self.latent_dimensions = latent_dimensions
        self.input_dim = input_dim

        self.verbose = True
        self.arch_name = "RL-Multi-VAE-50"

        self.success_weight = 1

        self.avg_loss_li = []
        self.total_loss_li = []

    def console_log(self, message):
        if not self.verbose:
            return
        print(message)

    @staticmethod
    def transmit_function(z_a):
        """
        identity transmission function
        """
        return z_a

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
        handles exploration of actions
        should be changed in subclasses
        in this case just the highest weight is chosen (no exploration)
        """
        chosen_mus = []
        chosen_log_vars = []
        chosen_indices = []

        for i in range(weights.shape[0]):
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
            # encode data point (encoder policy action)
            mus, logvar, weights = self.encoder_agent(x.to('cpu'))
            # determine which mean to use for each sample
            chosen_indices = torch.argmax(weights, dim=1)
            chosen_mus = torch.stack([mus[i, index] for i, index in enumerate(chosen_indices)])
            z = chosen_mus.detach()

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
        self.encoder_agent.to(self.device)

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
        plt.title('Average Loss Per Epoch')
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

            for x, _ in training_data_loader:
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
                reward = self.reward_function(x_a, x_b, chosen_mus, chosen_log_vars, weights.gather(1, chosen_indices.unsqueeze(1)))
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
                self.plot_latent(training_data_loader, f"images/{self.arch_name}-epoch-{epoch}-epsilon-{round(self.epsilon, 2)}-latent.png")

    def save_model(self, path):
        """
        save the encoder and decoder separately
        """
        time_now = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        torch.save(self.encoder_agent.state_dict(), f"{path}/{self.arch_name}-encoder-{time_now}.pth")
        torch.save(self.decoder_agent.state_dict(), f"{path}/{self.arch_name}-decoder-{time_now}.pth")
