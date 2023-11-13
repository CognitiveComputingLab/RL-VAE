import torch

from toy_data import data
from toy_data import plotting
from toy_data import embedding

from architectures import rl_vae, vae, ce_rl_vae, de_rl_vae
import helper

# module init
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device: ", device)


if __name__ == "__main__":
    # generate data to be embedded
    # toy_data = data.MoebiusStrip(n=10000, width=1, turns=6)
    toy_data = data.Circle2D(n=1000)
    toy_data.generate()
    toy_data.add_noise()
    input_dim = toy_data.data.shape[1]

    # plot the data
    # plotting.mpl_2d_plot(toy_data)

    # fig = plotting.scatter3d(toy_data)
    # fig.show()

    # use UMAP embedding
    umap = embedding.UMAP(toy_data)
    # umap.fit()
    # umap.plot()

    # train RL-VAE system on data
    model = rl_vae.RlVae(device, input_dim)
    model.success_weight = 1
    toy_dataset = helper.ToyTorchDataset(toy_data)
    data_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=128,
        shuffle=False
    )
    model.train(data_loader, epochs=100)
    model.plot_latent(data_loader, f"images/{model.arch_name}-latent.png")
    model.plot_loss(f"images/{model.arch_name}-loss.png")


