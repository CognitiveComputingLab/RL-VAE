import torch

from toy_data import data
from toy_data import plotting
from toy_data import embedding

from architectures import rl_vae, vae, ce_rl_vae
import helper

# module init
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device: ", device)

if __name__ == "__main__":
    # generate data to be embedded
    toy_data = data.MoebiusStrip(n=10000, width=1, turns=6)
    toy_data.generate()

    # plot the data
    fig = plotting.scatter3d(toy_data)
    # fig.show()

    # use UMAP embedding
    umap = embedding.UMAP(toy_data)
    # umap.fit()
    # umap.plot()

    # train RL-VAE system on data
    model = ce_rl_vae.ConstantExplorationRLVAE(device)
    model.exploration_rate = 0.5
    toy_dataset = helper.ToyTorchDataset(toy_data)
    data_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=128,
        shuffle=False
    )
    model.train(data_loader, epochs=100)
    model.plot_latent(data_loader, f"images/{model.arch_name}-latent.png")
    model.plot_loss(f"images/{model.arch_name}-loss.png")


