import datetime
import torch

from toy_data import data
from toy_data import plotting
from toy_data import embedding

from architectures import rl_vae, vae, ce_rl_vae, de_rl_vae, distance_rl_vae
import helper

# module init
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device: ", device)


if __name__ == "__main__":
    # generate data to be embedded
    toy_data = data.MoebiusStrip(n=10000, width=1, turns=1).generate()
    # toy_data = data.Sphere3D(10000).generate().add_noise(0.2)
    # toy_data = data.MusicData(10000).generate()
    input_dim = toy_data.data.shape[1]

    # plot the data
    # plotting.mpl_2d_plot(toy_data)
    # toy_data.plot()

    # fig = plotting.scatter3d(toy_data)
    # fig.show()

    # use UMAP embedding
    umap = embedding.UMAP(toy_data)
    # umap.fit()
    # umap.plot()

    # train RL-VAE system on data
    for i in [1]:
        # model = vae.VaeSystem(device, input_dim)
        # model = rl_vae.RlVae(device, input_dim)
        # model = de_rl_vae.DecreasingExplorationRLVAE(device, input_dim, 20)
        # model = ce_rl_vae.ConstantExplorationRLVAE(device, input_dim)
        model = distance_rl_vae.DistanceRLVAE(device, input_dim, 2)

        model.success_weight = i
        toy_dataset = helper.ToyTorchDataset(toy_data)
        data_loader = torch.utils.data.DataLoader(
            toy_dataset,
            batch_size=64,
            shuffle=False
        )
        try:
            model.train(data_loader, epochs=100)
        except KeyboardInterrupt:
            print("stopping early...")
        model.plot_latent(data_loader, f"images/{model.arch_name}-latent.png")
        model.plot_loss(f"images/{model.arch_name}-loss.png")
        # save model
        # model.save_model(f"models/")


