from architectures.EmbeddingFramework import EmbeddingFramework

from architectures.PropertyCalculators.PropertyCalculatorNone import PropertyCalculatorNone
from architectures.Samplers.SamplerVAE import SamplerVAE
from architectures.Encoders.EncoderVAE import EncoderVAE
from architectures.Explorers.ExplorerVAE import ExplorerVAE
from architectures.Decoders.DecoderSimple import DecoderSimple
from architectures.Transmitters.TransmitterIdentity import TransmitterIdentity
from architectures.RewardCalculators.RewardCalculatorVAE import RewardCalculatorVAE

from architectures.PropertyCalculators.PropertyCalculatorUMAP import PropertyCalculatorUMAP
from architectures.Samplers.SamplerUMAP import SamplerUMAP
from architectures.Encoders.EncoderSimple import EncoderSimple
from architectures.Explorers.ExplorerIdentity import ExplorerIdentity
from architectures.RewardCalculators.RewardCalculatorUMAP import RewardCalculatorUMAP


def preset_umap(device, input_dim, output_dim, data_loader):
    umap_embedding_framework = EmbeddingFramework(device)
    umap_embedding_framework.property_calculator = PropertyCalculatorUMAP(device, data_loader)
    umap_embedding_framework.sampler = SamplerUMAP(device, data_loader)
    umap_embedding_framework.encoder_agent = EncoderSimple(input_dim, output_dim).to(device)
    umap_embedding_framework.explorer = ExplorerIdentity(device)
    umap_embedding_framework.transmitter = TransmitterIdentity(device)
    umap_embedding_framework.decoder_agent = DecoderSimple(input_dim, output_dim).to(device)
    umap_embedding_framework.reward_calculator = RewardCalculatorUMAP(device)
    umap_embedding_framework.set_learning_mode(False)
    return umap_embedding_framework


def preset_vae(device, input_dim, output_dim, data_loader):
    vae_embedding_framework = EmbeddingFramework(device)
    vae_embedding_framework.property_calculator = PropertyCalculatorNone(device, data_loader)
    vae_embedding_framework.sampler = SamplerVAE(device, data_loader)
    vae_embedding_framework.encoder_agent = EncoderVAE(input_dim, output_dim).to(device)
    vae_embedding_framework.explorer = ExplorerVAE(device)
    vae_embedding_framework.transmitter = TransmitterIdentity(device)
    vae_embedding_framework.decoder_agent = DecoderSimple(input_dim, output_dim).to(device)
    vae_embedding_framework.reward_calculator = RewardCalculatorVAE(device)
    vae_embedding_framework.set_learning_mode(True)
    return vae_embedding_framework
