import torch
import numpy as np
from reverse_ldm import ReverseDDIM, ReverseSDE, ReverseDDPM
from hyper_param import HyperParamsDDIM, HyperParamsSDE, HyperParamsDDPM
from noise_predictor import NoisePredictor
from text_encoder import TextEncoder
from autoencoder import AutoencoderLDM
from sample_ldm import SampleLDM



def test_sample_ldm_conditional():

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    auto_encod = AutoencoderLDM(
        in_channels=3,
        down_channels=[8, 16],
        up_channels=[8, 16],
        out_channels=3,
        latent_channels=3,
        dropout_rate=0.1,
        num_heads=1,
        num_groups=8,
        num_layers_per_block=1,
        total_down_sampling_factor=2,
        use_vq=False,
        num_embeddings=16,
        beta=1e-4
    )
    noise_model = NoisePredictor(
        in_channels=3,
        down_channels=[16, 32],
        mid_channels=[32, 32],
        up_channels=[32, 16],
        down_sampling=[True, True],
        time_embed_dim=32,
        y_embed_dim=32,
        num_down_blocks=1,
        num_mid_blocks=1,
        num_up_blocks=1,
        dropout_rate=0.1,
        down_sampling_factor=2,
        where_y=True,
        y_to_all=False
    )
    conditional_model = TextEncoder(
        use_pretrained_model=True,
        model_name="bert-base-uncased",
        vocabulary_size=30522,
        num_layers=1,
        input_dimension=32,
        output_dimension=32,
        num_heads=1,
        context_length=77,
        dropout_rate=0.1,
        qkv_bias=False,
        scaling_value=4,
        epsilon=1e-5
    )
    #hyper_params = HyperParamsDDIM(num_steps=100, tau_num_steps=10)
    #hyper_params = HyperParamsDDPM(num_steps=100)
    hyper_params = HyperParamsSDE(num_steps=100)
    #reverse = ReverseDDIM(hyper_params)
    #reverse = ReverseDDPM(hyper_params)
    reverse = ReverseSDE(hyper_params, "ode")  # ve, vp, sub-vp, ode

    conditions = [
            "a cat sitting on a chair",
            "a dog running in the park"#,
            #"a sunny beach with palm trees",
            #"a snowy mountain landscape"
    ]

    generator = SampleLDM(
        model="sde",
        reverse_diffusion=reverse,
        noise_predictor=noise_model,
        compressor_model=auto_encod,
        image_shape=(128, 128),
        conditional_model=conditional_model,
        batch_size=2,
        in_channels=3,
        device=device
    )

    generated_imgs = generator(conditions=conditions)
    assert generated_imgs.shape == (2, 3, 128, 128), f"Expected shape (2, 3, 128, 128), got {generated_imgs.shape}"
    assert torch.all(generated_imgs >= 0) and torch.all(generated_imgs <= 1), "Images out of [0, 1] range"
    print("Conditional SampleLDM test passed!")



def test_sample_ldm_unconditional():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    auto_encod = AutoencoderLDM(
        in_channels=3,
        down_channels=[8, 16],
        up_channels=[8, 16],
        out_channels=3,
        latent_channels=3,
        dropout_rate=0.1,
        num_heads=1,
        num_groups=8,
        num_layers_per_block=1,
        total_down_sampling_factor=2,
        use_vq=False,
        num_embeddings=16,
        beta=1e-4
    )
    noise_model = NoisePredictor(
        in_channels=3,
        down_channels=[16, 32],
        mid_channels=[32, 32],
        up_channels=[32, 16],
        down_sampling=[True, True],
        time_embed_dim=32,
        y_embed_dim=32,
        num_down_blocks=1,
        num_mid_blocks=1,
        num_up_blocks=1,
        dropout_rate=0.1,
        down_sampling_factor=2,
        where_y=True,
        y_to_all=False
    )

    # hyper_params = HyperParamsDDIM(num_steps=100, tau_num_steps=10)
    #hyper_params = HyperParamsDDPM(num_steps=100)
    hyper_params = HyperParamsSDE(num_steps=100)
    # reverse = ReverseDDIM(hyper_params)
    # reverse = ReverseDDPM(hyper_params)
    reverse = ReverseSDE(hyper_params, "ode")  # ve, vp, sub-vp, ode

    generator = SampleLDM(
        model="sde",
        reverse_diffusion=reverse,
        noise_predictor=noise_model,
        compressor_model=auto_encod,
        image_shape=(128, 128),
        batch_size=2,
        in_channels=3,
        device=device
    )

    generated_imgs = generator()
    assert generated_imgs.shape == (2, 3, 128, 128), f"Expected shape (2, 3, 128, 128), got {generated_imgs.shape}"
    assert torch.all(generated_imgs >= 0) and torch.all(generated_imgs <= 1), "Images out of [0, 1] range"
    print("Unconditional SampleLDM test passed!")


if __name__ == "__main__":
    test_sample_ldm_conditional()
    test_sample_ldm_unconditional()