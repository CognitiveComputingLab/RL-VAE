import torch.nn as nn

import rl_embeddings.similarity_calculators as similarity_calculators
import rl_embeddings.samplers as samplers
import rl_embeddings.encoders as encoders
import rl_embeddings.explorers as explorers
import rl_embeddings.transmitters as transmitters
import rl_embeddings.decoders as decoders
import rl_embeddings.reward_calculators as reward_calculators


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

        # specifications
        self.reward_name = "total_reward"

    def forward(self, epoch=0):
        sampler_out = self.sampler()
        encoder_out = self.encoder(**sampler_out)
        explorer_out = self.explorer(**merge_dicts(encoder_out, {"epoch": epoch}))

        if not self.training:
            return explorer_out["encoded_points"], sampler_out

        decoder_out = self.decoder(**explorer_out)
        concat = merge_dicts(sampler_out, encoder_out, explorer_out, decoder_out)
        reward_out = self.reward(**concat)

        return reward_out, self.sampler.epoch_done


class KHeadVAE(VAE):
    def __init__(self, input_dim, latent_dim, device, data_loader, k=2):
        super(KHeadVAE, self).__init__(input_dim, latent_dim, device, data_loader)

        # components
        self.encoder = encoders.EncoderKHeadVAE(input_dim, latent_dim, k).to(device)
        self.explorer = explorers.ExplorerKHeadVAE(device)
        self.reward = reward_calculators.RewardCalculatorKHeadVAE(device)


class KHeadVAEDecreasing(KHeadVAE):
    def __init__(self, input_dim, latent_dim, device, data_loader, k=2):
        super().__init__(input_dim, latent_dim, device, data_loader, k)

        # change in component
        self.explorer = explorers.ExplorerKHeadVAEDecreasing(device)


class VarianceVAE(VAE):
    def __init__(self, input_dim, latent_dim, device, data_loader):
        super(VarianceVAE, self).__init__(input_dim, latent_dim, device, data_loader)

        # components
        self.encoder = encoders.EncoderSimple(input_dim, latent_dim).to(device)
        self.explorer = explorers.ExplorerVariance(device)
        self.reward = reward_calculators.RewardCalculatorMSE(device)


class VarianceVAEDecreasing(VarianceVAE):
    def __init__(self, input_dim, latent_dim, device, data_loader):
        super(VarianceVAE, self).__init__(input_dim, latent_dim, device, data_loader)

        # components
        self.encoder = encoders.EncoderSimple(input_dim, latent_dim).to(device)
        self.explorer = explorers.ExplorerVarianceDecreasing(device)


class UMAP(nn.Module):
    def __init__(self, input_dim, latent_dim, device, data_loader):
        super(UMAP, self).__init__()

        # components
        self.similarity = similarity_calculators.SimilarityCalculatorUMAP(device, data_loader)
        self.sampler = samplers.SamplerUMAP(device, data_loader)
        self.encoder = encoders.EncoderUMAP(input_dim, latent_dim).to(device)
        self.reward = reward_calculators.RewardCalculatorUMAP(device)

        # init high dim similarity
        self.similarities_initialized = False

        # specifications
        self.reward_name = "encoder_reward"

    def forward(self, epoch=0):
        # check high dim similarities
        if not self.similarities_initialized:
            self.similarity.calculate_high_dim_similarity()
            self.similarities_initialized = True

        sampler_out = self.sampler(**{"high_dim_similarity": self.similarity.high_dim_similarity})
        encoder_out = self.encoder(**sampler_out)

        if not self.training:
            return encoder_out["encoded_points"], sampler_out

        similarity_out = self.similarity(**merge_dicts(sampler_out, encoder_out))
        reward_out = self.reward(**merge_dicts(sampler_out, encoder_out, similarity_out))

        return reward_out, self.sampler.epoch_done


class TSNE(nn.Module):
    def __init__(self, input_dim, latent_dim, device, data_loader):
        super(TSNE, self).__init__()

        # components
        self.similarity = similarity_calculators.SimilarityCalculatorTSNE(device, data_loader)
        self.sampler = samplers.SamplerVAE(device, data_loader)
        self.encoder = encoders.EncoderSimple(input_dim, latent_dim).to(device)
        self.reward = reward_calculators.RewardCalculatorTSNE(device)

        # init high dim similarity
        self.similarities_initialized = False

        # specifications
        self.reward_name = "encoder_reward"

    def forward(self, epoch=0):
        # check high dim similarities
        if not self.similarities_initialized:
            self.similarity.calculate_high_dim_similarity()
            self.similarities_initialized = True

        sampler_out = self.sampler()
        encoder_out = self.encoder(**sampler_out)

        if not self.training:
            return encoder_out["encoded_points"], sampler_out

        similarity_out = self.similarity(**merge_dicts(sampler_out, encoder_out))
        reward_out = self.reward(**merge_dicts(sampler_out, encoder_out, similarity_out))

        return reward_out, self.sampler.epoch_done


class TSNE_UMAP(TSNE):
    def __init__(self, input_dim, latent_dim, device, data_loader):
        super(TSNE_UMAP, self).__init__(input_dim, latent_dim, device, data_loader)

        # new components
        self.similarity = similarity_calculators.SimilarityCalculatorTSNE_UMAP(device, data_loader)
