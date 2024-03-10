import torch
import helper
import examples
from toy_data import data
from tqdm import tqdm


def get_device():
    torch.manual_seed(0)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def train(emb_model, epochs, reward_name="total_reward"):
    optimizer = torch.optim.Adam(list(emb_model.parameters()))
    for epoch in tqdm(range(epochs), disable=False):
        emb_model.sampler.reset_epoch()

        # run through epoch
        epoch_done = False
        while not epoch_done:
            reward, epoch_done = emb_model.forward(epoch)
            loss = -reward[reward_name]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    device = get_device()

    toy_data = data.Sphere3D(n=1000).generate()
    toy_dataset = helper.ToyTorchDataset(toy_data)
    data_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=100,
        shuffle=False
    )

    model = examples.VAE(3, 2, device, data_loader)
    train(model, epochs=5)




