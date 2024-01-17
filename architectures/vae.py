import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils
import torch.distributions
import matplotlib.pyplot as plt

# module init
torch.manual_seed(0)


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 4096)
        self.linearM = nn.Linear(4096, latent_dims)
        self.linearS = nn.Linear(4096, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = functional.relu(self.linear1(x))
        x = functional.relu(self.linear2(x))
        x = functional.relu(self.linear3(x))
        # mean
        mu = self.linearM(x)
        # variance
        log_var = self.linearS(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 4096)
        self.linearOut = nn.Linear(4096, input_dim)

    def forward(self, z):
        z = functional.relu(self.linear1(z))
        z = functional.relu(self.linear2(z))
        z = functional.relu(self.linear3(z))
        z = self.linearOut(z)
        return z


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
    # get tensor of random values from normal distribution
    eps = torch.randn_like(std)
    # generate a sample
    return mu + std * eps


class VariationalAutoencoder(nn.Module):
    def __init__(self, device, input_dim, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dim, latent_dims).to(device)
        self.decoder = Decoder(input_dim, latent_dims).to(device)

    def forward(self, x):
        # encode
        mu, log_var = self.encoder(x)

        # re-parameterization
        z = re_parameterize(mu, log_var)

        # decode
        out = self.decoder(z)
        return out, mu, log_var


class VaeSystem:
    def __init__(self, device, input_dim, latent_dimensions=2):
        self.device = device
        self.latent_dimensions = latent_dimensions
        self.input_dim = input_dim
        self.autoencoder = VariationalAutoencoder(device, input_dim, self.latent_dimensions).to(device)
        self.optimiser = torch.optim.Adam(self.autoencoder.parameters())
        self.success_weight = 1000

        self.avg_loss_li = []
        self.total_loss_li = []

        self.arch_name = "VAE"

    def plot_loss(self, save_as):
        # Plotting the average loss per epoch
        plt.plot(range(len(self.avg_loss_li)), self.avg_loss_li)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Average Loss per Epoch')
        plt.savefig(save_as)
        plt.close()

    def plot_latent(self, data_loader, save_as, num_batches=100):
        self.autoencoder.encoder.to('cpu')
        for i, (x, y) in enumerate(data_loader):
            mu, log_var = self.autoencoder.encoder(x.to('cpu'))
            z = re_parameterize(mu, log_var)
            z = z.to('cpu').detach().numpy()
            colors = y[:, :3].to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=colors)
            if i >= num_batches:
                plt.colorbar()
                break
        plt.gca().set_aspect('equal', 'datalim')
        plt.savefig(save_as)
        plt.close()
        self.autoencoder.encoder.to(self.device)

    def train(self, data, epochs=100):

        for epoch in range(epochs):
            print(f"---------------------- EPOCH {epoch} ----------------------")
            total_loss = 0
            counter = 0
            for x, y in data:
                # get data
                x = x.to(self.device)

                # forward pass
                x_hat, mu, log_var = self.autoencoder(x)

                # get loss
                kl_divergence = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())
                loss = functional.mse_loss(x_hat, x, reduction='sum') * self.success_weight
                total_loss += float(loss)
                counter += 1

                # learn
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            # print epoch loss
            avg_loss = total_loss / counter
            self.total_loss_li.append(total_loss)
            self.avg_loss_li.append(avg_loss)
            print(f"total loss: {total_loss}")
            print(f"average loss: {avg_loss}")

            if epoch % 10 == 0:
                self.plot_latent(data, f"images/{self.arch_name}-epoch-{epoch}-latent.png")

    def save_model(self, save_as):
        torch.save(self.autoencoder.state_dict(), save_as)
