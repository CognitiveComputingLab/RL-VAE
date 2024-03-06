import rl_embeddings.property_calculators as property_calculators
import rl_embeddings.samplers as samplers
import rl_embeddings.encoders as encoders
import rl_embeddings.explorers as explorers
import rl_embeddings.transmitters as transmitters
import rl_embeddings.decoders as decoders
import rl_embeddings.reward_calculators as reward_calculators
from rl_embeddings.embedding_framework import EmbeddingFramework


def preset_umap(device, input_dim, output_dim, data_loader):
    ef = EmbeddingFramework(device)
    ef.property_calculator = property_calculators.PropertyCalculatorUMAP(device, data_loader)
    ef.sampler = samplers.SamplerUMAP(device, data_loader)
    ef.encoder_agent = encoders.EncoderUMAP(input_dim, output_dim).to(device)
    ef.explorer = explorers.ExplorerIdentity(device)
    ef.transmitter = transmitters.TransmitterIdentity(device)
    ef.decoder_agent = decoders.DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = reward_calculators.RewardCalculatorUMAP(device)
    return ef


def preset_tsne(device, input_dim, output_dim, data_loader):
    ef = EmbeddingFramework(device)
    ef.property_calculator = property_calculators.PropertyCalculatorTSNE(device, data_loader)
    ef.sampler = samplers.SamplerTSNE(device, data_loader)
    ef.encoder_agent = encoders.EncoderTSNE(input_dim, output_dim).to(device)
    ef.explorer = explorers.ExplorerIdentity(device)
    ef.transmitter = transmitters.TransmitterIdentity(device)
    ef.decoder_agent = decoders.DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = reward_calculators.RewardCalculatorTSNE(device)
    return ef


def preset_vae(device, input_dim, output_dim, data_loader):
    ef = EmbeddingFramework(device)
    ef.property_calculator = property_calculators.PropertyCalculatorNone(device, data_loader)
    ef.sampler = samplers.SamplerVAE(device, data_loader)
    ef.encoder_agent = encoders.EncoderVAE(input_dim, output_dim).to(device)
    ef.explorer = explorers.ExplorerVAE(device)
    ef.transmitter = transmitters.TransmitterIdentity(device)
    ef.decoder_agent = decoders.DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = reward_calculators.RewardCalculatorVAE(device)
    return ef


def preset_k_head_vae(device, input_dim, output_dim, data_loader, k=2):
    ef = EmbeddingFramework(device)
    ef.sampler = samplers.SamplerVAE(device, data_loader)
    ef.encoder_agent = encoders.EncoderKHeadVAE(input_dim, output_dim, k).to(device)
    ef.explorer = explorers.ExplorerKHeadVAEDecreasing(device)
    ef.property_calculator = property_calculators.PropertyCalculatorNone(device, data_loader)
    ef.transmitter = transmitters.TransmitterIdentity(device)
    ef.decoder_agent = decoders.DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = reward_calculators.RewardCalculatorKHeadVAE(device)
    return ef


def preset_variance_vae(device, input_dim, output_dim, data_loader):
    ef = EmbeddingFramework(device)
    ef.property_calculator = property_calculators.PropertyCalculatorNone(device, data_loader)
    ef.sampler = samplers.SamplerVAE(device, data_loader)
    ef.encoder_agent = encoders.EncoderSimple(input_dim, output_dim).to(device)
    ef.explorer = explorers.ExplorerVarianceDecreasing(device)
    ef.transmitter = transmitters.TransmitterIdentity(device)
    ef.decoder_agent = decoders.DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = reward_calculators.RewardCalculatorVarianceVAE(device)
    return ef
