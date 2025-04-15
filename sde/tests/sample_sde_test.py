import torch
from reverse_sde import ReverseSDE
from hyper_param import HyperParamsSDE
from text_encoder import TextEncoder
from noise_predictor import NoisePredictor
from sample_sde import SampleSDE
from transformers import BertTokenizer






def test_generate_ddpm_unconditional():
    device = torch.device("cuda")

    hyper_params = HyperParamsSDE(num_steps=100, beta_start=1e-4, beta_end=0.02, beta_method="linear",
                                  sigma_start=1e-3, sigma_end=1.0, start=0.0, end=1.0)
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
    generator = SampleSDE(
        method="sub-vp",  # ve, vp, sub-vp, ode
        noise_predictor=noise_predictor,
        hyper_params_model=hyper_params,
        image_shape=(32, 32),
        batch_size=2,
        in_channels=3,
        device=device
    )
    generated_imgs = generator()
    assert generated_imgs.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {generated_imgs.shape}"
    assert torch.all(generated_imgs >= 0) and torch.all(generated_imgs <= 1), "Images out of [0, 1] range"
    print("Unconditional GenerateDDPM test passed!")



def test_generate_ddpm_conditional():
    device = torch.device("cuda")

    hyper_params = HyperParamsSDE(num_steps=100, beta_start=1e-4, beta_end=0.02, beta_method="linear",
                                  sigma_start=1e-3, sigma_end=1.0, start=0.0, end=1.0)
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
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = TextEncoder(use_pretrained_model=True, model_name="bert-base-uncased", output_dimension=64)
    texts = ["A sunny beach", "A snowy mountain"]
    conditions = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77)["input_ids"]

    generator = SampleSDE(
        method="sub-vp",  # ve, vp, sub-vp, ode
        noise_predictor=noise_predictor,
        hyper_params_model=hyper_params,
        image_shape=(32, 32),
        conditions=conditions,
        conditional_model=text_encoder,
        batch_size=2,
        in_channels=3,
        device=device
    )

    generated_imgs = generator()
    assert generated_imgs.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {generated_imgs.shape}"
    assert torch.all(generated_imgs >= 0) and torch.all(generated_imgs <= 1), "Images out of [0, 1] range"
    print("Conditional GenerateDDPM test passed!")

if __name__ == "__main__":
    test_generate_ddpm_unconditional()
    test_generate_ddpm_conditional()
