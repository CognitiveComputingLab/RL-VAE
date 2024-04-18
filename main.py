"""
Example usage of rl-embedding framework
"""
import matplotlib.pyplot as plt
import torch
import examples
from toy_data import data, toy_torch_dataset
from tqdm import tqdm
import rl_embeddings.pre_trainers as pre_trainers


def get_device():
    torch.manual_seed(0)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class Main:
    def __init__(self, emb_model):
        self.emb_model = emb_model
        self.reward_history = []

    def train(self, epochs, latent_freq=10):
        optimizer = torch.optim.Adam(list(self.emb_model.parameters()), lr=0.001)
        for epoch in tqdm(range(epochs), disable=False):
            self.emb_model.sampler.reset_epoch()

            # run through epoch
            epoch_done = False
            epoch_reward = 0
            frame = 0
            while not epoch_done:
                reward, epoch_done = self.emb_model(epoch)
                loss = -reward[self.emb_model.reward_name]
                epoch_reward += float(reward[self.emb_model.reward_name])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                """frame += 1
                if frame == 5:
                    return"""

            self.reward_history.append(epoch_reward)

            if not latent_freq == 0 and epoch % latent_freq == 0:
                self.plot_latent(f"images/latent-{epoch}.png")
                # print(self.emb_model.explorer.current_exploration)

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

    def plot_reward(self, path):
        """
        plot the reward history of training the model
        """
        if len(self.reward_history) == 0:
            print("reward history is empty, aborting plot")
            return

        # remove first to increase visibility
        self.reward_history = self.reward_history[1:]

        # generating indices for x-axis
        indices = list(range(len(self.reward_history)))

        # generate image
        plt.plot(indices, self.reward_history, marker='o', linestyle='-', color='green')
        plt.title('Reward History Plot')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(path)
        plt.close()


def compare_umap(toy_data_obj):
    from toy_data.embedding import UMAP
    umap_obj = UMAP(toy_data_obj)
    umap_obj.fit()
    umap_obj.plot()


if __name__ == "__main__":
    # get pytorch device
    device = get_device()

    # initialise the dataset as a pytorch dataloader
    # toy_data = data.MoebiusStrip(turns=1, n=1000).generate()
    toy_data = data.FashionMNIST(n=1000).generate()
    toy_dataset = toy_torch_dataset.ToyTorchDataset(toy_data)
    data_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=100,
        shuffle=False
    )

    input_dim = toy_data.data.shape[1]
    latent_dim = 2

    # initialise the model
    model = examples.UMAP(input_dim, latent_dim, device, data_loader)
    # model = examples.TSNE(input_dim, latent_dim, device, data_loader)
    # model = examples.VAE(input_dim, latent_dim, device, data_loader)
    # model = examples.VarianceVAEDecreasing(input_dim, latent_dim, device, data_loader)
    # model = examples.KHeadVAEDecreasing(input_dim, latent_dim, device, data_loader, k=10)
    # model.explorer.current_exploration = 0
    # model.reward.success_weight = 100
    # model.reward.kl_weight = 0

    # Main
    m = Main(model)
    # m.plot_latent(f"images/no-training.png")

    # pretrain on spectral embedding
    # pre_trainer = pre_trainers.PreTrainerSpectral(model, device, data_loader)
    # pre_trainer.pre_train(epochs=100)
    # pre_trainer.plot_spectral("images/spectral.png")
    # m.plot_latent(f"images/pre-trained.png")

    # train the model
    m.train(epochs=500, latent_freq=20)
    m.plot_latent(f"images/post-training.png")
    m.plot_reward(f"images/reward-history.png")

