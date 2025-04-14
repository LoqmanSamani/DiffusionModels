import torch
from transformers import BertTokenizer
from hyper_param import HyperParamsDDIM
from reverse_ddim import ReverseDDIM
from noise_predictor import NoisePredictor
from text_encoder import TextEncoder
from sample_ddim import SampleDDIM



def test_generate_ddim_unconditional():
    """tests GenerateDDIM in unconditional mode."""

    hyper_params = HyperParamsDDIM(num_steps=100, tau_num_steps=10)
    noise_predictor = NoisePredictor(
        in_channels=3,
        down_channels=[64, 128],
        mid_channels=[128, 128],
        up_channels=[128, 64],
        down_sampling=[True, True],
        time_embed_dim=128,
        y_embed_dim=768,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2
    )
    generator = SampleDDIM(
        noise_predictor=noise_predictor,
        hyper_params_model=hyper_params,
        image_shape=(64, 64),
        batch_size=2,
        in_channels=3,
        device="cpu"
    )
    generated_imgs = generator()
    assert generated_imgs.shape == (2, 3, 64, 64), f"Expected shape (2, 3, 64, 64), got {generated_imgs.shape}"
    assert torch.all(generated_imgs >= 0) and torch.all(generated_imgs <= 1), "Images out of [0, 1] range"
    print("Unconditional GenerateDDPM test passed!")



def test_generate_ddim_conditional():
    """tests GenerateDDIM in conditional mode with TextEncoder."""
    hyper_params = HyperParamsDDIM(num_steps=100, tau_num_steps=10)

    noise_predictor = NoisePredictor(
        in_channels=3,
        down_channels=[64, 128],
        mid_channels=[128, 128],
        up_channels=[128, 64],
        down_sampling=[True, True],
        time_embed_dim=512,
        y_embed_dim=512,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout_rate=0.1
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = TextEncoder(use_pretrained_model=True, model_name="bert-base-uncased", output_dimension=512)
    texts = ["A sunny beach", "A snowy mountain"]
    conditions = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77)["input_ids"]

    generator = SampleDDIM(
        noise_predictor=noise_predictor,
        hyper_params_model=hyper_params,
        image_shape=(64, 64),
        conditions=conditions,
        conditional_model=text_encoder,
        batch_size=2,
        in_channels=3,
        device="cpu"
    )

    generated_imgs = generator()
    assert generated_imgs.shape == (2, 3, 64, 64), f"Expected shape (2, 3, 64, 64), got {generated_imgs.shape}"
    assert torch.all(generated_imgs >= 0) and torch.all(generated_imgs <= 1), "Images out of [0, 1] range"
    print("Conditional GenerateDDPM test passed!")

if __name__ == "__main__":
    test_generate_ddim_unconditional()
    test_generate_ddim_conditional()