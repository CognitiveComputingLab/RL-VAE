"""
Example usage of rl-embedding framework
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import examples
from toy_data import data, toy_torch_dataset
from tqdm import tqdm
import rl_embeddings.pre_trainers as pre_trainers


def get_device():
    torch.manual_seed(0)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class Main:
    def __init__(self, emb_model, toy_data_obj):
        self.emb_model = emb_model
        self.toy_data = toy_data_obj

        self.reward_history = []
        self.all_embeddings = []
        self.all_colors = []
        self.all_labels = []

    def train(self, epochs, latent_freq=10):
        optimizer = torch.optim.Adam(list(self.emb_model.parameters()), lr=0.001)
        for epoch in tqdm(range(epochs), disable=False):
            self.emb_model.sampler.reset_epoch()

            # run through epoch
            epoch_done = False
            epoch_reward = 0
            while not epoch_done:
                reward, epoch_done = self.emb_model(epoch)
                loss = -reward[self.emb_model.reward_name]
                epoch_reward += float(reward[self.emb_model.reward_name])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.reward_history.append(epoch_reward)

            if not latent_freq == 0 and epoch % latent_freq == 0:
                self.plot_latent(f"images/latent-{epoch}.png")
                # print(self.emb_model.explorer.current_exploration)

    def save_raw(self, path):
        # create arrays
        emb = np.array(self.all_embeddings)
        col = np.array(self.all_colors)
        rew = np.array(self.reward_history)
        lab = np.array(self.all_labels)

        # save to file
        np.savez(path, embeddings=emb, colors=col, rewards=rew, labels=lab, high_dim_data=self.toy_data.data)

        # return for further computation
        return emb, col, rew, lab

    def plot_latent(self, path):
        # init
        self.emb_model.eval()
        self.emb_model.sampler.reset_epoch()

        # collect full embeddings
        epoch_embeddings = []
        epoch_colors = []
        epoch_labels = []

        epoch_done = False
        while not epoch_done:
            # embed batch in eval mode
            z, y = self.emb_model()
            indices = y["indices"].to('cpu').numpy()
            labels = self.toy_data.labels[indices]
            epoch_labels.append(labels)
            y = y["points"][1]
            z = z.detach().to('cpu').numpy()

            # collect embeddings and colors for later use
            epoch_embeddings.append(z)
            colors = y[:, :3].to('cpu').detach().numpy()
            epoch_colors.append(colors)

            # plot batch of points
            plt.scatter(z[:, 0], z[:, 1], c=colors, s=15)

            # check if epoch is done
            epoch_done = self.emb_model.sampler.epoch_done

        # generate and save plot
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('Latent Projection')
        plt.savefig(path)
        plt.close()

        # save embeddings
        self.all_embeddings.append(np.vstack(epoch_embeddings))
        self.all_colors.append(np.vstack(epoch_colors))
        self.all_labels.append(np.concatenate(epoch_labels))

        # un-initialize
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
    umap_obj.fit(n_neighbors=20, min_dist=1)
    umap_obj.save_raw()
    umap_obj.plot()


if __name__ == "__main__":
    # get pytorch device
    device = get_device()

    # initialise the dataset as a pytorch dataloader
    toy_data = data.MoebiusStrip(turns=1, n=1000).generate()
    # toy_data = data.Sphere3D(n=1000).generate()
    toy_dataset = toy_torch_dataset.ToyTorchDataset(toy_data)
    data_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=100,
        shuffle=False
    )

    input_dim = toy_data.data.shape[1]
    latent_dim = 2
    print("data finished loading with shape: ", toy_data.data.shape)

    # initialise the model
    model = examples.KHeadVAEDecreasing(input_dim, latent_dim, device, data_loader, k=5)
    m = Main(model, toy_data)
    m.plot_latent(f"images/no-training.png")

    # pretrain on spectral embedding
    pre_trainer = pre_trainers.PreTrainerSpectral(model, device, data_loader)
    pre_trainer.pre_train(epochs=50)
    pre_trainer.plot_spectral("images/spectral.png")
    m.plot_latent(f"images/pre-trained.png")

    # train the model
    m.train(epochs=100, latent_freq=10)
    m.plot_reward(f"images/reward-history.png")
    em, co, re, la = m.save_raw(f"images/raw-data.npz")

