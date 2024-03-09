import torch
import helper
import presets
from toy_data import data
from rl_embeddings.property_calculators import PropertyCalculatorTSNE


def get_device():
    torch.manual_seed(0)
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def merge_dicts(*dicts):
    """
    merge multiple dictionaries into one
    later duplicate values overwrite earlier ones
    """
    merged_dict = {}
    for dictionary in dicts:
        merged_dict.update(dictionary)
    return merged_dict


if __name__ == "__main__":
    device = get_device()

    toy_data = data.Sphere3D(n=100).generate()
    toy_dataset = helper.ToyTorchDataset(toy_data)
    data_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=5,
        shuffle=False
    )

    import rl_embeddings.samplers as samplers
    import rl_embeddings.encoders as encoders
    import rl_embeddings.explorers as explorers
    import rl_embeddings.transmitters as transmitters
    import rl_embeddings.reward_calculators as reward_calculators
    import rl_embeddings.decoders as decoders

    s = samplers.SamplerVAE(device, data_loader)
    e = encoders.EncoderVAE(3, 2).to(device)
    ex = explorers.ExplorerVAE(device)
    t = transmitters.TransmitterIdentity(device)
    d = decoders.DecoderSimple(3, 2).to(device)
    r = reward_calculators.RewardCalculatorVAE(device)

    sampler_out = s()
    encoder_out = e(**sampler_out)
    explorer_out = ex(**encoder_out)
    transmitter_out = t(**explorer_out)
    decoder_out = d(**transmitter_out)
    concat = merge_dicts(sampler_out, encoder_out, explorer_out, transmitter_out, decoder_out)
    reward_out = r(**concat)

    print(reward_out)


    """embedding_framework = presets.preset_tsne(device, 3, 2, data_loader)
    embedding_framework.disable_tqdm = False
    embedding_framework.train_model(epochs=5, plot_interval=20)
    embedding_framework.plot_latent(f"images/latent.png")"""

    """
    TODO:
    - think of optimiser / reward system (done)
    - convert reward calculators (done)
    - convert embedding_framework class (done)
    - cleanup
        - eval (done)
        - tqdm (done)
        - etc.
    - implement t-sne
    - more generality in some of the classes?
    - compatibility check?
    - module
    """
