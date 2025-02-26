from clip import Clip
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from model_converter import load_from_standard_weights

def preload_model_from_standard_weights(model_weights_path,device):
    state_dict = load_from_standard_weights(model_weights_path,device)
    
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"],strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"],strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"],strict=True)

    clip = Clip().to(device)
    clip.load_state_dict(state_dict["clip"],strict=True)

    return {
        "clip":clip,
        "encoder":encoder,
        "decoder":decoder,
        "diffusion":diffusion,
    }

