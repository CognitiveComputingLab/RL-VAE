from architectures.EmbeddingFramework import EmbeddingFramework

import architectures.Encoders as Encoders
import architectures.Decoders as Decoders
import architectures.Explorers as Explorers
import architectures.PropertyCalculators as PropertyCalculators
import architectures.RewardCalculators as RewardCalculators
import architectures.Samplers as Samplers
import architectures.Transmitters as Transmitters


def preset_umap(device, input_dim, output_dim, data_loader):
    ef = EmbeddingFramework(device)
    ef.property_calculator = PropertyCalculators.PropertyCalculatorUMAP.PropertyCalculatorUMAP(device, data_loader)
    ef.sampler = Samplers.SamplerUMAP.SamplerUMAP(device, data_loader)
    ef.encoder_agent = Encoders.EncoderSimple.EncoderSimple(input_dim, output_dim).to(device)
    ef.explorer = Explorers.ExplorerIdentity.ExplorerIdentity(device)
    ef.transmitter = Transmitters.TransmitterIdentity.TransmitterIdentity(device)
    ef.decoder_agent = Decoders.DecoderSimple.DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = RewardCalculators.RewardCalculatorUMAP.RewardCalculatorUMAP(device)
    ef.set_learning_mode(encoder_reconstruction=False)
    return ef


def preset_vae(device, input_dim, output_dim, data_loader):
    ef = EmbeddingFramework(device)
    ef.property_calculator = PropertyCalculators.PropertyCalculatorNone.PropertyCalculatorNone(device, data_loader)
    ef.sampler = Samplers.SamplerVAE.SamplerVAE(device, data_loader)
    ef.encoder_agent = Encoders.EncoderVAE.EncoderVAE(input_dim, output_dim).to(device)
    ef.explorer = Explorers.ExplorerVAE.ExplorerVAE(device)
    ef.transmitter = Transmitters.TransmitterIdentity.TransmitterIdentity(device)
    ef.decoder_agent = Decoders.DecoderSimple.DecoderSimple(input_dim, output_dim).to(device)
    ef.reward_calculator = RewardCalculators.RewardCalculatorVAE.RewardCalculatorVAE(device)
    ef.set_learning_mode(encoder_reconstruction=True)
    return ef


def preset_k_head_vae(device, input_dim, output_dim, data_loader, k=2):
    ef = EmbeddingFramework(device)
    ef.sampler = Samplers.SamplerVAE.SamplerVAE(device, data_loader)
    ef.encoder_agent = Encoders.EncoderKHeadVAE.EncoderKHeadVAE(input_dim, output_dim, k)
    ef.explorer = Explorers.ExplorerKHeadVAE.ExplorerKHeadVAE(device)
    ef.property_calculator = PropertyCalculators.PropertyCalculatorNone.PropertyCalculatorNone(device, data_loader)
    ef.transmitter = Transmitters.TransmitterIdentity.TransmitterIdentity(device)
    ef.decoder_agent = Decoders.DecoderSimple.DecoderSimple(input_dim, output_dim)
    ef.reward_calculator = RewardCalculators.RewardCalculatorKHeadVAE.RewardCalculatorKHeadVAE(device)
    ef.set_learning_mode(encoder_reconstruction=True)
    return ef
