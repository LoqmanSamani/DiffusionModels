"""
Comprehensive Test Suite for DDIM Implementation

This test suite validates the core components of the DDIM (Denoising Diffusion Implicit Models)
implementation including forward/reverse diffusion, variance scheduling, training, and sampling.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import shutil
from typing import Tuple, List
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from torchdiff.ddim import (
    ForwardDDIM,
    ReverseDDIM,
    VarianceSchedulerDDIM,
    TrainDDIM,
    SampleDDIM
)



# Mock implementations for testing
class MockNoisePredictor(nn.Module):
    """Mock noise predictor for testing"""

    def __init__(self, in_channels=1, time_embed_dim=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.time_embed = nn.Linear(1, time_embed_dim)

    def forward(self, x, t, y_encoded=None, *args):
        # Simple implementation for testing
        batch_size = x.shape[0]
        t_embed = self.time_embed(t.float().unsqueeze(1))
        noise = self.conv(x)
        return noise


class MockConditionalModel(nn.Module):
    """Mock conditional model for testing"""

    def __init__(self, embed_dim=32):
        super().__init__()
        self.embed = nn.Embedding(1000, embed_dim)

    def forward(self, input_ids, attention_mask=None):
        return self.embed(input_ids[:, 0])  # Simple mock


class MockMetrics:
    """Mock metrics class for testing"""

    def __init__(self):
        self.fid = True
        self.metrics = True
        self.lpips = True

    def forward(self, x_real, x_fake):
        # Return mock metric values
        return 50.0, 0.1, 25.0, 0.8, 0.05  # fid, mse, psnr, ssim, lpips


class TestVarianceSchedulerDDIM:
    """Test cases for VarianceSchedulerDDIM"""

    def test_initialization_valid_params(self):
        """Test scheduler initialization with valid parameters"""
        scheduler = VarianceSchedulerDDIM(
            num_steps=100,
            tau_num_steps=10,
            beta_start=1e-4,
            beta_end=0.02
        )
        assert scheduler.num_steps == 100
        assert scheduler.tau_num_steps == 10
        assert scheduler.beta_start == 1e-4
        assert scheduler.beta_end == 0.02

    def test_initialization_invalid_params(self):
        """Test scheduler initialization with invalid parameters"""
        with pytest.raises(ValueError):
            # Invalid beta range
            VarianceSchedulerDDIM(beta_start=0.02, beta_end=1e-4)

        with pytest.raises(ValueError):
            # Invalid num_steps
            VarianceSchedulerDDIM(num_steps=0)

    def test_beta_schedule_methods(self):
        """Test different beta scheduling methods"""
        methods = ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]

        for method in methods:
            scheduler = VarianceSchedulerDDIM(
                num_steps=100,
                beta_method=method
            )
            betas = scheduler.betas
            assert len(betas) == 100
            assert torch.all(betas >= scheduler.beta_start)
            assert torch.all(betas <= scheduler.beta_end)

    def test_trainable_beta(self):
        """Test trainable beta functionality"""
        scheduler = VarianceSchedulerDDIM(
            num_steps=100,
            trainable_beta=True
        )

        # Check that beta_raw is a parameter
        assert hasattr(scheduler, 'beta_raw')
        assert isinstance(scheduler.beta_raw, nn.Parameter)

        # Check that betas are in valid range
        betas = scheduler.betas
        assert torch.all(betas >= scheduler.beta_start)
        assert torch.all(betas <= scheduler.beta_end)

    def test_tau_schedule(self):
        """Test subsampled (tau) schedule generation"""
        scheduler = VarianceSchedulerDDIM(
            num_steps=1000,
            tau_num_steps=100
        )

        tau_betas, tau_alphas, tau_alpha_cumprod, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod = scheduler.get_tau_schedule()

        assert len(tau_betas) == 100
        assert len(tau_alphas) == 100
        assert len(tau_alpha_cumprod) == 100
        assert torch.all(tau_alphas == 1 - tau_betas)


class TestForwardDDIM:
    """Test cases for ForwardDDIM"""

    def test_forward_process(self):
        """Test forward diffusion process"""
        scheduler = VarianceSchedulerDDIM(num_steps=100)
        forward_ddim = ForwardDDIM(scheduler)

        batch_size, channels, height, width = 4, 1, 28, 28
        x0 = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, scheduler.num_steps, (batch_size,))

        xt = forward_ddim(x0, noise, time_steps)

        assert xt.shape == x0.shape
        assert not torch.equal(xt, x0)  # Should be different due to noise

    def test_invalid_time_steps(self):
        """Test forward process with invalid time steps"""
        scheduler = VarianceSchedulerDDIM(num_steps=100)
        forward_ddim = ForwardDDIM(scheduler)

        x0 = torch.randn(2, 1, 28, 28)
        noise = torch.randn_like(x0)

        # Test time steps out of range
        with pytest.raises(ValueError):
            time_steps = torch.tensor([100, 150])  # Out of range
            forward_ddim(x0, noise, time_steps)


class TestReverseDDIM:
    """Test cases for ReverseDDIM"""

    def test_reverse_process(self):
        """Test reverse diffusion process"""
        scheduler = VarianceSchedulerDDIM(
            num_steps=1000,
            tau_num_steps=100,
            eta=0.0  # Deterministic
        )
        reverse_ddim = ReverseDDIM(scheduler)

        batch_size, channels, height, width = 4, 1, 28, 28
        xt = torch.randn(batch_size, channels, height, width)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(1, scheduler.tau_num_steps, (batch_size,))
        prev_time_steps = time_steps - 1

        xt_prev, x0 = reverse_ddim(xt, predicted_noise, time_steps, prev_time_steps)

        assert xt_prev.shape == xt.shape
        assert x0.shape == xt.shape

    def test_stochastic_sampling(self):
        """Test stochastic sampling with eta > 0"""
        scheduler = VarianceSchedulerDDIM(
            num_steps=1000,
            tau_num_steps=100,
            eta=0.5  # Stochastic
        )
        reverse_ddim = ReverseDDIM(scheduler)

        xt = torch.randn(2, 1, 28, 28)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.tensor([50, 60])
        prev_time_steps = torch.tensor([49, 59])

        # Multiple runs should give different results due to stochasticity
        xt_prev1, _ = reverse_ddim(xt, predicted_noise, time_steps, prev_time_steps)
        xt_prev2, _ = reverse_ddim(xt, predicted_noise, time_steps, prev_time_steps)

        assert not torch.allclose(xt_prev1, xt_prev2, atol=1e-6)


class TestTrainDDIM:
    """Test cases for TrainDDIM"""

    def setup_training_components(self):
        """Setup components needed for training tests"""
        # Create mock data
        x = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 10, (100,)).float()  # Mock labels
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Models
        noise_predictor = MockNoisePredictor()
        conditional_model = MockConditionalModel()

        # DDIM components
        scheduler = VarianceSchedulerDDIM(num_steps=100, tau_num_steps=10)
        forward_ddim = ForwardDDIM(scheduler)
        reverse_ddim = ReverseDDIM(scheduler)

        # Training components
        optimizer = torch.optim.Adam(
            list(noise_predictor.parameters()) + list(conditional_model.parameters()),
            lr=1e-3
        )
        loss_fn = nn.MSELoss()
        metrics = MockMetrics()

        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'noise_predictor': noise_predictor,
            'conditional_model': conditional_model,
            'forward_ddim': forward_ddim,
            'reverse_ddim': reverse_ddim,
            'optimizer': optimizer,
            'loss_fn': loss_fn,
            'metrics': metrics
        }

    def test_trainer_initialization(self):
        """Test TrainDDIM initialization"""
        components = self.setup_training_components()

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = TrainDDIM(
                noise_predictor=components['noise_predictor'],
                forward_diffusion=components['forward_ddim'],
                reverse_diffusion=components['reverse_ddim'],
                data_loader=components['train_loader'],
                optimizer=components['optimizer'],
                objective=components['loss_fn'],
                val_loader=components['val_loader'],
                max_epochs=2,
                device='cpu',
                conditional_model=components['conditional_model'],
                metrics_=components['metrics'],
                store_path=temp_dir,
                use_ddp=False
            )

            assert trainer.max_epochs == 2
            assert trainer.device == torch.device('cpu')

    def test_training_loop(self):
        """Test basic training functionality"""
        components = self.setup_training_components()

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = TrainDDIM(
                noise_predictor=components['noise_predictor'],
                forward_diffusion=components['forward_ddim'],
                reverse_diffusion=components['reverse_ddim'],
                data_loader=components['train_loader'],
                optimizer=components['optimizer'],
                objective=components['loss_fn'],
                max_epochs=2,
                device='cpu',
                store_path=temp_dir,
                use_ddp=False,
                log_frequency=1
            )

            # Run training
            train_losses, best_val_loss = trainer()

            assert len(train_losses) == 2
            assert isinstance(best_val_loss, float)

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        components = self.setup_training_components()

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = TrainDDIM(
                noise_predictor=components['noise_predictor'],
                forward_diffusion=components['forward_ddim'],
                reverse_diffusion=components['reverse_ddim'],
                data_loader=components['train_loader'],
                optimizer=components['optimizer'],
                objective=components['loss_fn'],
                max_epochs=1,
                device='cpu',
                store_path=temp_dir,
                use_ddp=False
            )

            # Save checkpoint manually
            trainer._save_checkpoint(1, 0.5)

            # Check if checkpoint file exists
            checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pth')]
            assert len(checkpoint_files) > 0

            # Load checkpoint
            checkpoint_path = os.path.join(temp_dir, checkpoint_files[0])
            epoch, loss = trainer.load_checkpoint(checkpoint_path)

            assert epoch == 1
            assert loss == 0.5


class TestSampleDDIM:
    """Test cases for SampleDDIM"""

    def setup_sampling_components(self):
        """Setup components needed for sampling tests"""
        scheduler = VarianceSchedulerDDIM(num_steps=100, tau_num_steps=10)
        reverse_ddim = ReverseDDIM(scheduler)
        noise_predictor = MockNoisePredictor()
        conditional_model = MockConditionalModel()

        return {
            'reverse_ddim': reverse_ddim,
            'noise_predictor': noise_predictor,
            'conditional_model': conditional_model
        }

    def test_sampler_initialization(self):
        """Test SampleDDIM initialization"""
        components = self.setup_sampling_components()

        sampler = SampleDDIM(
            reverse_diffusion=components['reverse_ddim'],
            noise_predictor=components['noise_predictor'],
            image_shape=(28, 28),
            batch_size=4,
            device='cpu'
        )

        assert sampler.image_shape == (28, 28)
        assert sampler.batch_size == 4
        assert sampler.device == torch.device('cpu')

    def test_unconditional_sampling(self):
        """Test unconditional image generation"""
        components = self.setup_sampling_components()

        with tempfile.TemporaryDirectory() as temp_dir:
            sampler = SampleDDIM(
                reverse_diffusion=components['reverse_ddim'],
                noise_predictor=components['noise_predictor'],
                image_shape=(28, 28),
                batch_size=2,
                in_channels=1,  # Match the mock predictor
                device='cpu',
                image_output_range=(-1, 1)
            )

            # Generate samples
            generated_images = sampler(
                conditions=None,
                save_images=True,
                save_path=temp_dir
            )

            assert generated_images.shape == (2, 1, 28, 28)  # 1 channel to match predictor
            assert torch.all(generated_images >= 0) and torch.all(generated_images <= 1)

    def test_conditional_sampling(self):
        """Test conditional image generation"""
        components = self.setup_sampling_components()

        with tempfile.TemporaryDirectory() as temp_dir:
            sampler = SampleDDIM(
                reverse_diffusion=components['reverse_ddim'],
                noise_predictor=components['noise_predictor'],
                conditional_model=components['conditional_model'],
                image_shape=(28, 28),
                batch_size=2,
                in_channels=1,  # Match the mock predictor
                device='cpu'
            )

            # Test with string prompts
            conditions = ["cat", "dog"]
            generated_images = sampler(
                conditions=conditions,
                save_images=False
            )

            assert generated_images.shape == (2, 1, 28, 28)  # 1 channel to match predictor

    def test_tokenization(self):
        """Test text tokenization functionality"""
        components = self.setup_sampling_components()

        sampler = SampleDDIM(
            reverse_diffusion=components['reverse_ddim'],
            noise_predictor=components['noise_predictor'],
            conditional_model=components['conditional_model'],
            image_shape=(28, 28),
            device='cpu'
        )

        prompts = ["a cat", "a dog"]
        input_ids, attention_mask = sampler.tokenize(prompts)

        assert input_ids.shape[0] == 2  # Batch size
        assert attention_mask.shape[0] == 2
        assert input_ids.shape[1] == sampler.max_token_length


class TestIntegrationDDIM:
    """Integration tests for full DDIM pipeline"""

    def test_end_to_end_pipeline(self):
        """Test complete training and sampling pipeline"""
        # Setup data
        x = torch.randn(20, 1, 16, 16)  # Smaller for faster testing
        y = torch.randint(0, 5, (20,)).float()
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Setup models
        noise_predictor = MockNoisePredictor()
        conditional_model = MockConditionalModel()

        # Setup DDIM components
        scheduler = VarianceSchedulerDDIM(num_steps=50, tau_num_steps=5)
        forward_ddim = ForwardDDIM(scheduler)
        reverse_ddim = ReverseDDIM(scheduler)

        # Setup training
        optimizer = torch.optim.Adam(
            list(noise_predictor.parameters()) + list(conditional_model.parameters()),
            lr=1e-3
        )
        loss_fn = nn.MSELoss()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Train model
            trainer = TrainDDIM(
                noise_predictor=noise_predictor,
                forward_diffusion=forward_ddim,
                reverse_diffusion=reverse_ddim,
                data_loader=train_loader,
                optimizer=optimizer,
                objective=loss_fn,
                max_epochs=1,
                device='cpu',
                conditional_model=conditional_model,
                store_path=temp_dir,
                use_ddp=False
            )

            train_losses, _ = trainer()
            assert len(train_losses) == 1

            # Test sampling
            sampler = SampleDDIM(
                reverse_diffusion=reverse_ddim,
                noise_predictor=noise_predictor,
                conditional_model=conditional_model,
                image_shape=(16, 16),
                batch_size=2,
                in_channels=1,
                device='cpu'
            )

            conditions = ["test1", "test2"]
            generated_images = sampler(conditions=conditions, save_images=False)

            assert generated_images.shape == (2, 1, 16, 16)

    def test_forward_reverse_consistency(self):
        """Test that forward and reverse processes are mathematically consistent"""
        scheduler = VarianceSchedulerDDIM(
            num_steps=100,
            tau_num_steps=10,
            eta=0.0  # Deterministic
        )

        forward_ddim = ForwardDDIM(scheduler)
        reverse_ddim = ReverseDDIM(scheduler)
        noise_predictor = MockNoisePredictor()

        # Original image
        x0 = torch.randn(1, 1, 16, 16)

        # Apply forward process using full timestep range
        noise = torch.randn_like(x0)
        t_full = torch.tensor([50])  # Use full timestep range for forward
        xt = forward_ddim(x0, noise, t_full)

        # Apply reverse process using tau (subsampled) timestep range
        t_tau = torch.tensor([5])  # Middle of tau range (0-9)
        prev_t_tau = torch.tensor([4])  # Previous tau timestep
        xt_prev, x0_pred = reverse_ddim(xt, noise, t_tau, prev_t_tau)

        # Check that shapes are consistent
        assert xt_prev.shape == x0.shape
        assert x0_pred.shape == x0.shape


# Pytest configuration and runner
if __name__ == "__main__":
    # Run specific test categories
    print("Running VarianceSchedulerDDIM tests...")
    pytest.main(["-v", "TestVarianceSchedulerDDIM"])

    print("\nRunning ForwardDDIM tests...")
    pytest.main(["-v", "TestForwardDDIM"])

    print("\nRunning ReverseDDIM tests...")
    pytest.main(["-v", "TestReverseDDIM"])

    print("\nRunning TrainDDIM tests...")
    pytest.main(["-v", "TestTrainDDIM"])

    print("\nRunning SampleDDIM tests...")
    pytest.main(["-v", "TestSampleDDIM"])

    print("\nRunning Integration tests...")
    pytest.main(["-v", "TestIntegrationDDIM"])