import torch
from hyper_param import HyperParamsDDIM
from reverse_ddim import ReverseDDIM
from noise_predictor import NoisePredictor
from text_encoder import TextEncoder
from sample_ddim import SampleDDIM





def test_sample_ddim_unconditional():
    device = torch.device("cuda")

    hyper_params = HyperParamsDDIM(num_steps=1000, tau_num_steps=100, beta_start=1e-4, beta_end=0.02, beta_method="linear")
    noise_predictor = NoisePredictor(
        in_channels=3,
        down_channels=[32, 64],
        mid_channels=[64, 64],
        up_channels=[64, 32],
        down_sampling=[True, True],
        time_embed_dim=64,
        y_embed_dim=64,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout_rate=0.1
    )
    reverse = ReverseDDIM(hyper_params)
    generator = SampleDDIM(
        reverse_diffusion=reverse,
        noise_predictor=noise_predictor,
        image_shape=(32, 32),
        batch_size=2,
        in_channels=3,
        device=device
    )
    generated_imgs = generator()
    assert generated_imgs.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {generated_imgs.shape}"
    assert torch.all(generated_imgs >= 0) and torch.all(generated_imgs <= 1), "Images out of [0, 1] range"
    print("Unconditional SampleDDIM test passed!")



def  test_sample_ddim_conditional():
    device = torch.device("cuda")

    hyper_params = HyperParamsDDIM(num_steps=100, beta_start=1e-4, beta_end=0.02, beta_method="linear")
    noise_predictor = NoisePredictor(
        in_channels=3,
        down_channels=[32, 64],
        mid_channels=[64, 64],
        up_channels=[64, 32],
        down_sampling=[True, True],
        time_embed_dim=64,
        y_embed_dim=64,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout_rate=0.1
    )
    reverse = ReverseDDIM(hyper_params)

    prompts = ["a cat sitting on a chair", "a dog running in the park"]
    text_encoder = TextEncoder(use_pretrained_model=True, model_name="bert-base-uncased", output_dimension=64)
    generator = SampleDDIM(
        reverse_diffusion=reverse,
        noise_predictor=noise_predictor,
        conditional_model=text_encoder,
        image_shape=(32, 32),
        batch_size=2,
        in_channels=3,
        device=device
    )

    generated_imgs = generator(conditions=prompts)
    assert generated_imgs.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {generated_imgs.shape}"
    assert torch.all(generated_imgs >= 0) and torch.all(generated_imgs <= 1), "Images out of [0, 1] range"
    print("Conditional SampleDDIM test passed!")

if __name__ == "__main__":
    test_sample_ddim_unconditional()
    test_sample_ddim_conditional()