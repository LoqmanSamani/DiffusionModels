import torch
import torch.nn as nn
import unittest
from PIL import Image
import os
import numpy as np
from typing import Optional
from torchdiff.unclip import(
    VarianceSchedulerUnCLIP, ForwardUnCLIP, ReverseUnCLIP,
    CLIPEncoder, CLIPContextProjection, CLIPEmbeddingProjection,
    UnCLIPTransformerPrior, TrainUnCLIPPrior,
    UnClipDecoder, TrainUnClipDecoder,
    UpsamplerUnCLIP, TrainUpsamplerUnCLIP,
    SampleUnCLIP
)


# Mock Noise Predictor for UnClipDecoder and UpsamplerUnCLIP
class MockNoisePredictor(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None,
                clip_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.conv(x)


# Mock GLIDE Text Encoder
class MockGLIDETextEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, max_length: int = 77):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        return torch.randn(batch_size, self.max_length, self.embedding_dim, device=input_ids.device)


# Mock Metric (Loss Function)
class MockMetric:
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)


# Mock DataLoader
class MockDataLoader:
    def __init__(self, batch_size: int, low_res_size: int = 64, high_res_size: int = 256):
        self.batch_size = batch_size
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size

    def __iter__(self):
        return self

    def __next__(self):
        low_res_images = torch.randn(self.batch_size, 3, self.low_res_size, self.low_res_size)
        high_res_images = torch.randn(self.batch_size, 3, self.high_res_size, self.high_res_size)
        return low_res_images, high_res_images


class TestUnCLIP(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")  # Use CPU for testing to avoid CUDA dependency
        self.batch_size = 2
        self.clip_embedding_dim = 512
        self.image_size = (3, 64, 64)
        self.high_res_size = 256
        self.tau_num_steps = 10
        self.num_steps = 50

        # Initialize variance scheduler
        self.variance_scheduler = VarianceSchedulerUnCLIP(
            num_steps=self.num_steps,
            tau_num_steps=self.tau_num_steps,
            beta_start=1e-4,
            beta_end=0.02,
            beta_method="linear"
        ).to(self.device)

        # Initialize forward and reverse diffusion
        self.forward_diffusion = ForwardUnCLIP(self.variance_scheduler).to(self.device)
        self.reverse_diffusion = ReverseUnCLIP(self.variance_scheduler, prediction_type="noise").to(self.device)

        # Initialize mock components
        self.noise_predictor = MockNoisePredictor(in_channels=3, out_channels=3).to(self.device)
        self.glide_text_encoder = MockGLIDETextEncoder(embedding_dim=self.clip_embedding_dim).to(self.device)
        self.metric = MockMetric()

    def test_variance_scheduler_unclip(self):
        # Test initialization
        self.assertEqual(self.variance_scheduler.num_steps, self.num_steps)
        self.assertEqual(self.variance_scheduler.tau_num_steps, self.tau_num_steps)
        self.assertEqual(self.variance_scheduler.betas.shape, (self.num_steps,))
        self.assertTrue(torch.all(self.variance_scheduler.betas >= 1e-4))
        self.assertTrue(torch.all(self.variance_scheduler.betas <= 0.02))

        # Test beta schedule computation
        betas = self.variance_scheduler.compute_beta_schedule((1e-4, 0.02), self.num_steps, "linear")
        self.assertEqual(betas.shape, (self.num_steps,))
        self.assertTrue(torch.all(betas >= 1e-4))
        self.assertTrue(torch.all(betas <= 0.02))

        # Test tau schedule
        tau_betas, tau_alphas, tau_alpha_cumprod, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod = \
            self.variance_scheduler.get_tau_schedule()
        self.assertEqual(tau_betas.shape, (self.tau_num_steps,))
        self.assertEqual(tau_alphas.shape, (self.tau_num_steps,))
        self.assertEqual(tau_alpha_cumprod.shape, (self.tau_num_steps,))

    def test_forward_unclip(self):
        # Test forward diffusion
        x0 = torch.randn(self.batch_size, *self.image_size).to(self.device)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, self.num_steps, (self.batch_size,), device=self.device)
        xt = self.forward_diffusion(x0, noise, time_steps)
        self.assertEqual(xt.shape, x0.shape)
        self.assertTrue(torch.all(torch.isfinite(xt)))

        # Test 2D input (latent embeddings)
        x0_2d = torch.randn(self.batch_size, self.clip_embedding_dim).to(self.device)
        noise_2d = torch.randn_like(x0_2d)
        xt_2d = self.forward_diffusion(x0_2d, noise_2d, time_steps)
        self.assertEqual(xt_2d.shape, x0_2d.shape)

        # Test invalid time_steps
        with self.assertRaises(ValueError):
            invalid_time_steps = torch.tensor([self.num_steps], device=self.device)
            self.forward_diffusion(x0, noise, invalid_time_steps)

    def test_reverse_unclip(self):
        # Test reverse diffusion (noise prediction)
        xt = torch.randn(self.batch_size, *self.image_size).to(self.device)
        model_prediction = torch.randn_like(xt)
        time_steps = torch.randint(0, self.tau_num_steps, (self.batch_size,), device=self.device)
        prev_time_steps = torch.max(time_steps - 1, torch.tensor(0, device=self.device))
        xt_prev, x0 = self.reverse_diffusion(xt, model_prediction, time_steps, prev_time_steps)
        self.assertEqual(xt_prev.shape, xt.shape)
        self.assertEqual(x0.shape, xt.shape)
        self.assertTrue(torch.all(torch.isfinite(xt_prev)))
        self.assertTrue(torch.all(torch.isfinite(x0)))

        # Test x0 prediction
        self.reverse_diffusion.set_prediction_type("x0")
        xt_prev, x0 = self.reverse_diffusion(xt, model_prediction, time_steps, prev_time_steps)
        self.assertEqual(xt_prev.shape, xt.shape)
        self.assertEqual(x0.shape, xt.shape)

        # Test invalid time_steps
        with self.assertRaises(ValueError):
            invalid_time_steps = torch.tensor([self.tau_num_steps], device=self.device)
            self.reverse_diffusion(xt, model_prediction, invalid_time_steps, prev_time_steps)

    def test_clip_encoder(self):
        # Initialize CLIPEncoder with a mocked CLIP model
        class MockCLIPModel(nn.Module):
            def __init__(self):
                super().__init__()

            def get_image_features(self, pixel_values):
                return torch.randn(pixel_values.shape[0], self.clip_embedding_dim)

            def get_text_features(self, input_ids, attention_mask=None):
                return torch.randn(input_ids.shape[0], self.clip_embedding_dim)

        class MockCLIPProcessor:
            def __init__(self):
                pass

            def __call__(self, images=None, text=None, return_tensors="pt", padding=True, truncation=True):
                if images:
                    return {"pixel_values": torch.randn(len(images), 3, 224, 224)}
                if text:
                    return {"input_ids": torch.randint(0, 1000, (len(text), 77)),
                            "attention_mask": torch.ones(len(text), 77)}

        clip_encoder = CLIPEncoder(model_name="mock", device=self.device)
        clip_encoder.model = MockCLIPModel().to(self.device)
        clip_encoder.processor = MockCLIPProcessor()

        # Test image encoding
        images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in
                  range(self.batch_size)]
        image_embeddings = clip_encoder(images, data_type="img")
        self.assertEqual(image_embeddings.shape, (self.batch_size, self.clip_embedding_dim))

        # Test text encoding
        texts = ["A test prompt"] * self.batch_size
        text_embeddings = clip_encoder(texts, data_type="text")
        self.assertEqual(text_embeddings.shape, (self.batch_size, self.clip_embedding_dim))

        # Test similarity
        similarity = clip_encoder.compute_similarity(image_embeddings, text_embeddings)
        self.assertEqual(similarity.shape, (self.batch_size, self.batch_size))

    def test_unclip_decoder(self):
        # Initialize UnClipDecoder
        decoder = UnClipDecoder(
            clip_embedding_dim=self.clip_embedding_dim,
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_diffusion,
            reverse_diffusion=self.reverse_diffusion,
            glide_text_encoder=self.glide_text_encoder,
            device=self.device
        )

        # Test forward pass
        image_embeddings = torch.randn(self.batch_size, self.clip_embedding_dim).to(self.device)
        text_embeddings = torch.randn(self.batch_size, self.clip_embedding_dim).to(self.device)
        images = torch.randn(self.batch_size, *self.image_size).to(self.device)
        texts = ["test prompt"] * self.batch_size
        predicted_noise, noise = decoder(image_embeddings, text_embeddings, images, texts, p_classifier_free=0.0,
                                         p_text_drop=0.0)
        self.assertEqual(predicted_noise.shape, images.shape)
        self.assertEqual(noise.shape, images.shape)

        # Test classifier-free guidance
        modified_embeddings = decoder._apply_classifier_free_guidance(image_embeddings, p_value=0.05)
        self.assertEqual(modified_embeddings.shape, image_embeddings.shape)

        # Test text dropout
        dropped_embeddings = decoder._apply_text_dropout(text_embeddings, p_value=0.6)
        self.assertIsNone(dropped_embeddings)

    def test_unclip_transformer_prior(self):
        # Initialize UnCLIPTransformerPrior
        prior = UnCLIPTransformerPrior(
            forward_diffusion=self.forward_diffusion,
            reverse_diffusion=self.reverse_diffusion,
            clip_text_projection=None,
            clip_image_projection=None,
            transformer_embedding_dim=self.clip_embedding_dim
        ).to(self.device)

        # Test forward pass
        text_embeddings = torch.randn(self.batch_size, self.clip_embedding_dim).to(self.device)
        noisy_image_embeddings = torch.randn(self.batch_size, self.clip_embedding_dim).to(self.device)
        timesteps = torch.randint(0, self.num_steps, (self.batch_size,), device=self.device)
        predicted_embeddings = prior(text_embeddings, noisy_image_embeddings, timesteps)
        self.assertEqual(predicted_embeddings.shape, (self.batch_size, self.clip_embedding_dim))

    def test_clip_context_projection(self):
        # Initialize CLIPContextProjection
        projection = CLIPContextProjection(clip_embedding_dim=self.clip_embedding_dim, num_tokens=4).to(self.device)

        # Test forward pass
        z_i = torch.randn(self.batch_size, self.clip_embedding_dim).to(self.device)
        c = projection(z_i)
        self.assertEqual(c.shape, (self.batch_size, 4, self.clip_embedding_dim))

    def test_clip_embedding_projection(self):
        # Initialize CLIPEmbeddingProjection
        projection = CLIPEmbeddingProjection(
            clip_embedding_dim=self.clip_embedding_dim,
            transformer_embedding_dim=320
        ).to(self.device)

        # Test forward and inverse transform
        x = torch.randn(self.batch_size, self.clip_embedding_dim).to(self.device)
        x_reduced = projection(x)
        self.assertEqual(x_reduced.shape, (self.batch_size, 320))
        x_reconstructed = projection.inverse_transform(x_reduced)
        self.assertEqual(x_reconstructed.shape, x.shape)

        # Test reconstruction loss
        loss = projection.reconstruction_loss(x)
        self.assertTrue(torch.isfinite(loss))

    def test_upsampler_unclip(self):
        # Initialize UpsamplerUnCLIP
        upsampler = UpsamplerUnCLIP(
            forward_diffusion=self.forward_diffusion,
            reverse_diffusion=self.reverse_diffusion,
            in_channels=3,
            out_channels=3,
            model_channels=64,
            num_res_blocks=2,
            low_res_size=64,
            high_res_size=256
        ).to(self.device)

        # Test forward pass
        x_high = torch.randn(self.batch_size, 3, 256, 256).to(self.device)
        t = torch.randint(0, self.tau_num_steps, (self.batch_size,), device=self.device)
        x_low = torch.randn(self.batch_size, 3, 64, 64).to(self.device)
        predicted_noise = upsampler(x_high, t, x_low)
        self.assertEqual(predicted_noise.shape, (self.batch_size, 3, 256, 256))

    def test_train_upsampler_unclip(self):
        # Initialize TrainUpsamplerUnCLIP
        upsampler = UpsamplerUnCLIP(
            forward_diffusion=self.forward_diffusion,
            reverse_diffusion=self.reverse_diffusion,
            in_channels=3,
            out_channels=3,
            model_channels=64,
            num_res_blocks=2
        ).to(self.device)
        train_loader = MockDataLoader(batch_size=self.batch_size)
        optimizer = torch.optim.Adam(upsampler.parameters(), lr=1e-3)
        trainer = TrainUpsamplerUnCLIP(
            upsampler_model=upsampler,
            train_loader=train_loader,
            optimizer=optimizer,
            objective=self.metric,
            max_epochs=1,
            device=self.device,
            use_ddp=False,
            use_autocast=False
        )

        # Test training
        train_losses, best_val_loss = trainer()
        self.assertTrue(len(train_losses) == 1)
        self.assertTrue(isinstance(best_val_loss, float))

    def test_sample_unclip(self):
        # Initialize SampleUnCLIP
        prior = UnCLIPTransformerPrior(
            forward_diffusion=self.forward_diffusion,
            reverse_diffusion=self.reverse_diffusion,
            transformer_embedding_dim=self.clip_embedding_dim
        ).to(self.device)
        decoder = UnClipDecoder(
            clip_embedding_dim=self.clip_embedding_dim,
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_diffusion,
            reverse_diffusion=self.reverse_diffusion,
            glide_text_encoder=self.glide_text_encoder,
            device=self.device
        )
        clip_encoder = CLIPEncoder(model_name="mock", device=self.device)
        clip_encoder.model = MockCLIPModel().to(self.device)
        clip_encoder.processor = MockCLIPProcessor()
        upsampler = UpsamplerUnCLIP(
            forward_diffusion=self.forward_diffusion,
            reverse_diffusion=self.reverse_diffusion,
            in_channels=3,
            out_channels=3,
            model_channels=64
        ).to(self.device)

        sample_unclip = SampleUnCLIP(
            prior_model=prior,
            decoder_model=decoder,
            clip_model=clip_encoder,
            low_res_upsampler=upsampler,
            second_upsampler_model=None,
            device=self.device,
            batch_size=self.batch_size
        )

        # Test full pipeline
        prompts = ["A test image"] * self.batch_size
        final_images = sample_unclip(prompts=prompts, save_images=False)
        self.assertEqual(final_images.shape, (self.batch_size, 3, 256, 256))
        self.assertTrue(torch.all(torch.isfinite(final_images)))

        # Verify saved images (optional)
        final_images = sample_unclip(prompts=prompts, save_images=True, save_path="test_output")
        self.assertTrue(os.path.exists(os.path.join("test_output", "images_256", "image_1.png")))


if __name__ == "__main__":
    unittest.main()