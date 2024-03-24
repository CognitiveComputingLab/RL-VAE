"""
Example usage of rl-embedding framework
"""
import matplotlib.pyplot as plt
import torch
import examples
from toy_data import data, toy_torch_dataset
from tqdm import tqdm


def get_device():
    torch.manual_seed(0)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class Main:
    def __init__(self, emb_model):
        self.emb_model = emb_model

    def train(self, epochs, latent_freq=10):
        optimizer = torch.optim.Adam(list(self.emb_model.parameters()))
        for epoch in tqdm(range(epochs), disable=False):
            self.emb_model.sampler.reset_epoch()

            # run through epoch
            epoch_done = False
            while not epoch_done:
                reward, epoch_done = self.emb_model(epoch)
                loss = -reward[self.emb_model.reward_name]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % latent_freq == 0:
                self.plot_latent(f"images/latent-{epoch}.png")

    def plot_latent(self, path):
        # init
        self.emb_model.eval()
        self.emb_model.sampler.reset_epoch()

        epoch_done = False
        while not epoch_done:
            # embed batch in eval mode
            z, y = self.emb_model()
            z = z.detach().to('cpu').numpy()

            # break loop when epoch done
            epoch_done = self.emb_model.sampler.epoch_done

            # plot batch of points
            colors = y[:, :3].to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=colors)

        # generate image
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'latent projection')
        plt.savefig(path)
        plt.close()

        # un-init
        self.emb_model.train()


if __name__ == "__main__":
    # get pytorch device
    device = get_device()

    # initialise the dataset as a pytorch dataloader
    toy_data = data.MoebiusStrip(turns=1, n=1000).generate()
    toy_dataset = toy_torch_dataset.ToyTorchDataset(toy_data)
    data_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=100,
        shuffle=False
    )

    # initialise the model
    # model = examples.UMAP(3, 2, device, data_loader)
    # model = examples.TSNE(3, 2, device, data_loader)
    model = examples.VAE(3, 2, device, data_loader)
    # model = examples.VarianceVAE(3, 2, device, data_loader)
    model.reward.success_weight = 100
    # model = examples.KHeadVAEDecreasing(3, 2, device, data_loader, k=5)

    # train the model
    m = Main(model)
    m.train(epochs=11, latent_freq=5)

