import torch.nn as nn

import rl_embeddings.property_calculators as property_calculators
import rl_embeddings.samplers as samplers
import rl_embeddings.encoders as encoders
import rl_embeddings.explorers as explorers
import rl_embeddings.transmitters as transmitters
import rl_embeddings.decoders as decoders
import rl_embeddings.reward_calculators as reward_calculators
from rl_embeddings.embedding_framework import EmbeddingFramework


def merge_dicts(*dicts):
    """
    merge multiple dictionaries into one
    later duplicate values overwrite earlier ones
    """
    merged_dict = {}
    for dictionary in dicts:
        merged_dict.update(dictionary)
    return merged_dict


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, device, data_loader):
        super(VAE, self).__init__()

        # components
        self.sampler = samplers.SamplerVAE(device, data_loader)
        self.encoder = encoders.EncoderVAE(input_dim, latent_dim).to(device)
        self.explorer = explorers.ExplorerVAE(device)
        self.decoder = decoders.DecoderSimple(input_dim, latent_dim).to(device)
        self.reward = reward_calculators.RewardCalculatorVAE(device)

    def forward(self, epoch=0):
        sampler_out = self.sampler()
        encoder_out = self.encoder(**sampler_out)
        explorer_out = self.explorer(**encoder_out)
        decoder_out = self.decoder(**explorer_out)
        concat = merge_dicts(sampler_out, encoder_out, explorer_out, decoder_out)
        reward_out = self.reward(**concat)

        return reward_out, self.sampler.epoch_done
