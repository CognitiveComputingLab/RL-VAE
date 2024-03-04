from architectures.embedding_framework import EmbeddingFramework

from architectures.property_calculators import PropertyCalculatorUMAP
from architectures.samplers import SamplerUMAP
from architectures.encoders import EncoderUMAP
from architectures.encoders import EncoderSimple
from architectures.explorers import ExplorerIdentity
from architectures.transmitters import TransmitterIdentity
from architectures.Decoders.DecoderSimple import DecoderSimple
from architectures.RewardCalculators.RewardCalculatorUMAP import RewardCalculatorUMAP

from architectures.property_calculators import PropertyCalculatorNone
from architectures.samplers import SamplerVAE
from architectures.encoders import EncoderVAE
from architectures.explorers import ExplorerVAE
from architectures.RewardCalculators.RewardCalculatorVAE import RewardCalculatorVAE

from architectures.encoders import EncoderKHeadVAE
from architectures.explorers import ExplorerKHeadVAE, ExplorerKHeadVAEDecreasing
from architectures.RewardCalculators.RewardCalculatorKHeadVAE import RewardCalculatorKHeadVAE

from architectures.explorers import ExplorerVarianceDecreasing


def preset_umap(device, input_dim, output_dim, data_loader):
    ef = EmbeddingFramework(device)
    ef.property_calculator = PropertyCalculatorUMAP(device, data_loader)
    ef.sampler = SamplerUMAP(device, data_loader)
    ef.encoder_agent = EncoderUMAP(input_dim, output_dim).to(device)
    ef.explorer = ExplorerIdentity(device)
    ef.transmitter = TransmitterIdentity(device)
    ef.decoder_agent = DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = RewardCalculatorUMAP(device)
    ef.set_learning_mode(encoder_reconstruction=False)
    return ef


def preset_vae(device, input_dim, output_dim, data_loader):
    ef = EmbeddingFramework(device)
    ef.property_calculator = PropertyCalculatorNone(device, data_loader)
    ef.sampler = SamplerVAE(device, data_loader)
    ef.encoder_agent = EncoderVAE(input_dim, output_dim).to(device)
    ef.explorer = ExplorerVAE(device)
    ef.transmitter = TransmitterIdentity(device)
    ef.decoder_agent = DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = RewardCalculatorVAE(device)
    ef.set_learning_mode(encoder_reconstruction=True)
    return ef


def preset_k_head_vae(device, input_dim, output_dim, data_loader, k=2):
    ef = EmbeddingFramework(device)
    ef.sampler = SamplerVAE(device, data_loader)
    ef.encoder_agent = EncoderKHeadVAE(input_dim, output_dim, k).to(device)
    ef.explorer = ExplorerKHeadVAEDecreasing(device)
    ef.property_calculator = PropertyCalculatorNone(device, data_loader)
    ef.transmitter = TransmitterIdentity(device)
    ef.decoder_agent = DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = RewardCalculatorKHeadVAE(device)
    ef.set_learning_mode(encoder_reconstruction=True)
    return ef


def preset_variance_vae(device, input_dim, output_dim, data_loader):
    ef = EmbeddingFramework(device)
    ef.property_calculator = PropertyCalculatorNone(device, data_loader)
    ef.sampler = SamplerVAE(device, data_loader)
    ef.encoder_agent = EncoderSimple(input_dim, output_dim).to(device)
    ef.explorer = ExplorerVarianceDecreasing(device)
    ef.transmitter = TransmitterIdentity(device)
    ef.decoder_agent = DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = RewardCalculatorUMAP(device)
    ef.set_learning_mode(encoder_reconstruction=True)
    return ef
