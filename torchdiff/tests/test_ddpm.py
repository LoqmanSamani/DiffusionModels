"""
Comprehensive test suite for the DDPM (Denoising Diffusion Probabilistic Models) implementation.

This test suite covers all major components of the DDPM implementation including:
- VarianceSchedulerDDPM (noise scheduling)
- ForwardDDPM (forward diffusion process)
- ReverseDDPM (reverse diffusion process)
- TrainDDPM (training loop)
- SampleDDPM (image generation)
- Integration tests with different configurations

Usage:
    python test_ddpm.py
"""

import unittest
import torch
import torch.nn as nn
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import warnings
from torchdiff.ddpm import VarianceSchedulerDDPM, ForwardDDPM, ReverseDDPM, TrainDDPM, SampleDDPM



# Simple mock classes for testing
class MockNoisePredictor(nn.Module):
    """Simple noise predictor for testing"""

    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, in_channels, 3, padding=1)
        self.time_mlp = nn.Linear(1, 16)

    def forward(self, x, t, y_encoded=None, context=None):
        # Simple time embedding
        t_embed = self.time_mlp(t.float().unsqueeze(-1))
        t_embed = t_embed.view(t_embed.shape[0], t_embed.shape[1], 1, 1)
        t_embed = t_embed.expand(-1, -1, x.shape[2], x.shape[3])

        x = self.conv1(x)
        x = x + t_embed
        x = torch.relu(x)
        x = self.conv2(x)
        return x


class MockConditionalModel(nn.Module):
    """Simple conditional model for testing"""

    def __init__(self, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        return torch.randn(batch_size, self.embed_dim)


class MockMetrics:
    """Mock metrics class for testing"""

    def __init__(self):
        self.fid = True
        self.metrics = True
        self.lpips = True

    def forward(self, real_imgs, fake_imgs):
        # Return mock metrics
        return 50.0, 0.1, 25.0, 0.8, 0.05  # fid, mse, psnr, ssim, lpips


class TestVarianceSchedulerDDPM(unittest.TestCase):
    """Test cases for VarianceSchedulerDDPM"""

    def setUp(self):
        self.num_steps = 100
        self.beta_start = 1e-4
        self.beta_end = 0.02

    def test_initialization_fixed_schedule(self):
        """Test initialization with fixed (non-trainable) schedule"""
        scheduler = VarianceSchedulerDDPM(
            num_steps=self.num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            trainable_beta=False
        )

        self.assertEqual(scheduler.num_steps, self.num_steps)
        self.assertEqual(scheduler.beta_start, self.beta_start)
        self.assertEqual(scheduler.beta_end, self.beta_end)
        self.assertFalse(scheduler.trainable_beta)

        # Check that buffers are registered
        self.assertTrue(hasattr(scheduler, 'betas'))
        self.assertTrue(hasattr(scheduler, 'alphas'))
        self.assertTrue(hasattr(scheduler, 'alpha_bars'))
        self.assertEqual(scheduler.betas.shape[0], self.num_steps)

    def test_initialization_trainable_schedule(self):
        """Test initialization with trainable schedule"""
        scheduler = VarianceSchedulerDDPM(
            num_steps=self.num_steps,
            trainable_beta=True
        )

        self.assertTrue(scheduler.trainable_beta)
        self.assertIsInstance(scheduler.betas, nn.Parameter)

    def test_invalid_parameters(self):
        """Test initialization with invalid parameters"""
        with self.assertRaises(ValueError):
            VarianceSchedulerDDPM(beta_start=0.02, beta_end=1e-4)  # start > end

        with self.assertRaises(ValueError):
            VarianceSchedulerDDPM(num_steps=-1)  # negative steps

        with self.assertRaises(ValueError):
            VarianceSchedulerDDPM(beta_start=0.0)  # zero start

    def test_beta_schedule_methods(self):
        """Test different beta schedule computation methods"""
        methods = ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]

        for method in methods:
            scheduler = VarianceSchedulerDDPM(
                num_steps=self.num_steps,
                beta_method=method
            )
            self.assertEqual(scheduler.betas.shape[0], self.num_steps)
            self.assertTrue(torch.all(scheduler.betas >= self.beta_start))
            self.assertTrue(torch.all(scheduler.betas <= self.beta_end))

    def test_invalid_beta_method(self):
        """Test invalid beta schedule method"""
        with self.assertRaises(ValueError):
            VarianceSchedulerDDPM(beta_method="invalid_method")

    def test_compute_schedule_fixed(self):
        """Test compute_schedule method with fixed schedule"""
        scheduler = VarianceSchedulerDDPM(num_steps=self.num_steps, trainable_beta=False)

        # Test without time_steps (all steps)
        betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars = scheduler.compute_schedule()

        self.assertEqual(betas.shape[0], self.num_steps)
        self.assertEqual(alphas.shape[0], self.num_steps)
        self.assertEqual(alpha_bars.shape[0], self.num_steps)

        # Test with specific time_steps
        time_steps = torch.tensor([0, 50, 99])
        result = scheduler.compute_schedule(time_steps)
        betas_t, alphas_t, alpha_bars_t, sqrt_alpha_bars_t, sqrt_one_minus_alpha_bars_t = result

        self.assertEqual(betas_t.shape[0], 3)
        self.assertEqual(alphas_t.shape[0], 3)

    def test_compute_schedule_trainable(self):
        """Test compute_schedule method with trainable schedule"""
        scheduler = VarianceSchedulerDDPM(num_steps=self.num_steps, trainable_beta=True)

        time_steps = torch.tensor([0, 50, 99])
        result = scheduler.compute_schedule(time_steps)
        betas_t, alphas_t, alpha_bars_t, sqrt_alpha_bars_t, sqrt_one_minus_alpha_bars_t = result

        self.assertEqual(betas_t.shape[0], 3)
        self.assertTrue(torch.all(betas_t > 0))
        self.assertTrue(torch.all(betas_t < 1))


class TestForwardDDPM(unittest.TestCase):
    """Test cases for ForwardDDPM"""

    def setUp(self):
        self.variance_scheduler = VarianceSchedulerDDPM(num_steps=100, trainable_beta=False)
        self.forward_ddpm = ForwardDDPM(self.variance_scheduler)
        self.batch_size = 4
        self.channels = 1
        self.height = 28
        self.width = 28

    def test_forward_process(self):
        """Test the forward diffusion process"""
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, self.variance_scheduler.num_steps, (self.batch_size,))

        xt = self.forward_ddpm(x0, noise, time_steps)

        # Check output shape
        self.assertEqual(xt.shape, x0.shape)

        # Check that output is finite
        self.assertTrue(torch.all(torch.isfinite(xt)))

    def test_forward_process_trainable(self):
        """Test forward process with trainable variance scheduler"""
        variance_scheduler = VarianceSchedulerDDPM(num_steps=100, trainable_beta=True)
        forward_ddpm = ForwardDDPM(variance_scheduler)

        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, variance_scheduler.num_steps, (self.batch_size,))

        xt = forward_ddpm(x0, noise, time_steps)
        self.assertEqual(xt.shape, x0.shape)

    def test_invalid_time_steps(self):
        """Test with invalid time steps"""
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)

        # Time steps out of range
        invalid_time_steps = torch.tensor([100, 200, -1, 50])

        with self.assertRaises(ValueError):
            self.forward_ddpm(x0, noise, invalid_time_steps)

    def test_edge_cases(self):
        """Test edge cases like t=0 and t=max"""
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)

        # Test t=0 (should be close to original)
        time_steps = torch.zeros(self.batch_size, dtype=torch.long)
        xt = self.forward_ddpm(x0, noise, time_steps)
        self.assertEqual(xt.shape, x0.shape)

        # Test t=max-1
        time_steps = torch.full((self.batch_size,), self.variance_scheduler.num_steps - 1, dtype=torch.long)
        xt = self.forward_ddpm(x0, noise, time_steps)
        self.assertEqual(xt.shape, x0.shape)


class TestReverseDDPM(unittest.TestCase):
    """Test cases for ReverseDDPM"""

    def setUp(self):
        self.variance_scheduler = VarianceSchedulerDDPM(num_steps=100, trainable_beta=False)
        self.reverse_ddpm = ReverseDDPM(self.variance_scheduler)
        self.batch_size = 4
        self.channels = 1
        self.height = 28
        self.width = 28

    def test_reverse_process(self):
        """Test the reverse diffusion process"""
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(1, self.variance_scheduler.num_steps, (self.batch_size,))

        xt_minus_1 = self.reverse_ddpm(xt, predicted_noise, time_steps)

        # Check output shape
        self.assertEqual(xt_minus_1.shape, xt.shape)

        # Check that output is finite
        self.assertTrue(torch.all(torch.isfinite(xt_minus_1)))

    def test_reverse_process_t_zero(self):
        """Test reverse process at t=0 (deterministic)"""
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.zeros(self.batch_size, dtype=torch.long)

        xt_minus_1 = self.reverse_ddpm(xt, predicted_noise, time_steps)

        # At t=0, the process should be deterministic (no random noise added)
        self.assertEqual(xt_minus_1.shape, xt.shape)

    def test_reverse_process_trainable(self):
        """Test reverse process with trainable variance scheduler"""
        variance_scheduler = VarianceSchedulerDDPM(num_steps=100, trainable_beta=True)
        reverse_ddpm = ReverseDDPM(variance_scheduler)

        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(1, variance_scheduler.num_steps, (self.batch_size,))

        xt_minus_1 = reverse_ddpm(xt, predicted_noise, time_steps)
        self.assertEqual(xt_minus_1.shape, xt.shape)

    def test_invalid_time_steps(self):
        """Test with invalid time steps"""
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        predicted_noise = torch.randn_like(xt)

        # Time steps out of range
        invalid_time_steps = torch.tensor([100, 200, -1, 50])

        with self.assertRaises(ValueError):
            self.reverse_ddpm(xt, predicted_noise, invalid_time_steps)


class TestTrainDDPM(unittest.TestCase):
    """Test cases for TrainDDPM"""

    def setUp(self):
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()

        # Create mock data
        self.batch_size = 8
        self.channels = 1
        self.height = 28
        self.width = 28

        # Create mock dataset
        x_data = torch.randn(32, self.channels, self.height, self.width)
        y_data = ['test prompt'] * 32
        dataset = TensorDataset(x_data, torch.arange(32))  # Simple labels

        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Create components
        self.variance_scheduler = VarianceSchedulerDDPM(num_steps=50, trainable_beta=False)
        self.forward_ddpm = ForwardDDPM(self.variance_scheduler)
        self.reverse_ddpm = ReverseDDPM(self.variance_scheduler)
        self.noise_predictor = MockNoisePredictor(self.channels)
        self.optimizer = torch.optim.Adam(self.noise_predictor.parameters(), lr=1e-3)
        self.objective = nn.MSELoss()

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Test TrainDDPM initialization"""
        trainer = TrainDDPM(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_ddpm,
            reverse_diffusion=self.reverse_ddpm,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            max_epochs=2,
            device='cpu',
            store_path=self.test_dir
        )

        self.assertEqual(trainer.max_epochs, 2)
        self.assertEqual(trainer.device, torch.device('cpu'))
        self.assertIsNotNone(trainer.scheduler)
        self.assertIsNotNone(trainer.warmup_lr_scheduler)

    def test_training_loop_basic(self):
        """Test basic training loop without validation"""
        trainer = TrainDDPM(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_ddpm,
            reverse_diffusion=self.reverse_ddpm,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            max_epochs=2,
            device='cpu',
            store_path=self.test_dir,
            log_frequency=1
        )

        train_losses, best_val_loss = trainer()

        self.assertEqual(len(train_losses), 2)  # 2 epochs
        self.assertTrue(all(isinstance(loss, float) for loss in train_losses))
        self.assertIsInstance(best_val_loss, float)

    def test_training_with_validation(self):
        """Test training loop with validation"""
        metrics = MockMetrics()

        trainer = TrainDDPM(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_ddpm,
            reverse_diffusion=self.reverse_ddpm,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            val_loader=self.val_loader,
            metrics_=metrics,
            max_epochs=2,
            device='cpu',
            store_path=self.test_dir,
            val_frequency=1,
            log_frequency=1
        )

        train_losses, best_val_loss = trainer()

        self.assertEqual(len(train_losses), 2)
        self.assertIsInstance(best_val_loss, float)

    def test_training_with_conditional_model(self):
        """Test training with conditional model"""
        conditional_model = MockConditionalModel()

        # Modify data loader to return string labels
        x_data = torch.randn(16, self.channels, self.height, self.width)
        y_data = ['test prompt'] * 16

        # Create custom dataset that returns proper string prompts
        class StringDataset:
            def __init__(self, x_data, y_data):
                self.x_data = x_data
                self.y_data = y_data

            def __len__(self):
                return len(self.x_data)

            def __getitem__(self, idx):
                return self.x_data[idx], self.y_data[idx]

        string_dataset = StringDataset(x_data, y_data)
        string_loader = DataLoader(string_dataset, batch_size=4, shuffle=True)

        trainer = TrainDDPM(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_ddpm,
            reverse_diffusion=self.reverse_ddpm,
            data_loader=string_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            conditional_model=conditional_model,
            max_epochs=1,
            device='cpu',
            store_path=self.test_dir,
            log_frequency=1
        )

        train_losses, best_val_loss = trainer()
        self.assertEqual(len(train_losses), 1)

    def test_gradient_accumulation(self):
        """Test training with gradient accumulation"""
        trainer = TrainDDPM(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_ddpm,
            reverse_diffusion=self.reverse_ddpm,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            max_epochs=1,
            device='cpu',
            store_path=self.test_dir,
            grad_accumulation_steps=2,
            log_frequency=1
        )

        train_losses, best_val_loss = trainer()
        self.assertEqual(len(train_losses), 1)

    def test_warmup_scheduler(self):
        """Test warmup scheduler functionality"""
        optimizer = torch.optim.Adam(self.noise_predictor.parameters(), lr=1e-3)
        warmup_epochs = 5

        scheduler = TrainDDPM.warmup_scheduler(optimizer, warmup_epochs)

        # Test the lambda function directly
        lr_lambda = scheduler.lr_lambdas[0]

        # Test warmup phase - lr should increase linearly from 0 to 1
        for epoch in range(warmup_epochs):
            expected_lr = epoch / warmup_epochs
            actual_lr = lr_lambda(epoch)
            self.assertAlmostEqual(actual_lr, expected_lr, places=5)

        # Test post-warmup phase - should be 1.0
        self.assertEqual(lr_lambda(warmup_epochs), 1.0)
        self.assertEqual(lr_lambda(warmup_epochs + 1), 1.0)
        self.assertEqual(lr_lambda(warmup_epochs + 10), 1.0)  # Much later

        # Test edge cases
        self.assertEqual(lr_lambda(0), 0.0)  # At start, lr multiplier should be 0
        self.assertAlmostEqual(lr_lambda(warmup_epochs - 1), (warmup_epochs - 1) / warmup_epochs, places=5)

        # Test with different warmup_epochs value
        scheduler2 = TrainDDPM.warmup_scheduler(optimizer, 10)
        lr_lambda2 = scheduler2.lr_lambdas[0]

        self.assertEqual(lr_lambda2(0), 0.0)
        self.assertAlmostEqual(lr_lambda2(5), 0.5, places=5)
        self.assertEqual(lr_lambda2(10), 1.0)

    @patch('torch.save')
    def test_checkpoint_saving(self, mock_save):
        """Test checkpoint saving functionality"""
        trainer = TrainDDPM(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_ddpm,
            reverse_diffusion=self.reverse_ddpm,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            max_epochs=1,
            device='cpu',
            store_path=self.test_dir,
            val_frequency=1
        )

        # Test private checkpoint saving method
        trainer._save_checkpoint(1, 0.5)

        # Verify torch.save was called
        self.assertTrue(mock_save.called)

    def test_checkpoint_loading(self):
        """Test checkpoint loading functionality"""
        trainer = TrainDDPM(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_ddpm,
            reverse_diffusion=self.reverse_ddpm,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            max_epochs=1,
            device='cpu',
            store_path=self.test_dir
        )

        # Create a mock checkpoint
        checkpoint = {
            'epoch': 5,
            'model_state_dict_noise_predictor': self.noise_predictor.state_dict(),
            'model_state_dict_conditional': None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': 0.5,
            'variance_scheduler_model': self.variance_scheduler.state_dict(),
            'max_epochs': 10,
        }

        checkpoint_path = os.path.join(self.test_dir, 'test_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        # Test loading
        epoch, loss = trainer.load_checkpoint(checkpoint_path)

        self.assertEqual(epoch, 5)
        self.assertEqual(loss, 0.5)

    def test_validation_method(self):
        """Test validation method"""
        metrics = MockMetrics()

        trainer = TrainDDPM(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_ddpm,
            reverse_diffusion=self.reverse_ddpm,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            val_loader=self.val_loader,
            metrics_=metrics,
            max_epochs=1,
            device='cpu',
            store_path=self.test_dir
        )

        # Call validation
        val_metrics = trainer.validate()

        self.assertEqual(len(val_metrics), 6)  # val_loss, fid, mse, psnr, ssim, lpips
        val_loss, fid, mse, psnr, ssim, lpips_score = val_metrics
        self.assertIsInstance(val_loss, float)


class TestSampleDDPM(unittest.TestCase):
    """Test cases for SampleDDPM"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        self.variance_scheduler = VarianceSchedulerDDPM(num_steps=50, trainable_beta=False)
        self.reverse_ddpm = ReverseDDPM(self.variance_scheduler)
        self.noise_predictor = MockNoisePredictor(in_channels=3)
        self.conditional_model = MockConditionalModel()

        self.image_shape = (32, 32)
        self.batch_size = 4

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Test SampleDDPM initialization"""
        sampler = SampleDDPM(
            reverse_diffusion=self.reverse_ddpm,
            noise_predictor=self.noise_predictor,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            device='cpu'
        )

        self.assertEqual(sampler.image_shape, self.image_shape)
        self.assertEqual(sampler.batch_size, self.batch_size)
        self.assertEqual(sampler.device, torch.device('cpu'))

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters"""
        with self.assertRaises(ValueError):
            SampleDDPM(
                reverse_diffusion=self.reverse_ddpm,
                noise_predictor=self.noise_predictor,
                image_shape=(32,),  # Invalid shape
                batch_size=self.batch_size
            )

        with self.assertRaises(ValueError):
            SampleDDPM(
                reverse_diffusion=self.reverse_ddpm,
                noise_predictor=self.noise_predictor,
                image_shape=self.image_shape,
                batch_size=0  # Invalid batch size
            )

    def test_tokenization(self):
        """Test text tokenization"""
        sampler = SampleDDPM(
            reverse_diffusion=self.reverse_ddpm,
            noise_predictor=self.noise_predictor,
            image_shape=self.image_shape,
            conditional_model=self.conditional_model,
            batch_size=self.batch_size,
            device='cpu'
        )

        # Test single prompt
        prompts = "a beautiful landscape"
        input_ids, attention_mask = sampler.tokenize(prompts)

        self.assertEqual(input_ids.shape[0], 1)
        self.assertEqual(attention_mask.shape[0], 1)

        # Test multiple prompts
        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        input_ids, attention_mask = sampler.tokenize(prompts)

        self.assertEqual(input_ids.shape[0], 3)
        self.assertEqual(attention_mask.shape[0], 3)

    def test_unconditional_generation(self):
        """Test unconditional image generation"""
        sampler = SampleDDPM(
            reverse_diffusion=self.reverse_ddpm,
            noise_predictor=self.noise_predictor,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            device='cpu'
        )

        generated_imgs = sampler(
            conditions=None,
            normalize_output=True,
            save_images=False
        )

        expected_shape = (self.batch_size, 3, self.image_shape[0], self.image_shape[1])
        self.assertEqual(generated_imgs.shape, expected_shape)

        # Check that images are normalized to [0, 1]
        self.assertTrue(torch.all(generated_imgs >= 0))
        self.assertTrue(torch.all(generated_imgs <= 1))

    def test_conditional_generation(self):
        """Test conditional image generation"""
        sampler = SampleDDPM(
            reverse_diffusion=self.reverse_ddpm,
            noise_predictor=self.noise_predictor,
            image_shape=self.image_shape,
            conditional_model=self.conditional_model,
            batch_size=self.batch_size,
            device='cpu'
        )

        prompts = ["a beautiful sunset", "a mountain landscape", "a city skyline", "a forest path"]

        generated_imgs = sampler(
            conditions=prompts,
            normalize_output=True,
            save_images=False
        )

        expected_shape = (self.batch_size, 3, self.image_shape[0], self.image_shape[1])
        self.assertEqual(generated_imgs.shape, expected_shape)

    def test_image_saving(self):
        """Test image saving functionality"""
        sampler = SampleDDPM(
            reverse_diffusion=self.reverse_ddpm,
            noise_predictor=self.noise_predictor,
            image_shape=self.image_shape,
            batch_size=2,  # Use smaller batch for faster test
            device='cpu'
        )

        generated_imgs = sampler(
            conditions=None,
            save_images=True,
            save_path=self.test_dir
        )

        # Check that images were saved
        saved_files = os.listdir(self.test_dir)
        self.assertEqual(len(saved_files), 2)  # 2 images
        self.assertTrue(all(f.endswith('.png') for f in saved_files))

    def test_device_transfer(self):
        """Test device transfer functionality"""
        sampler = SampleDDPM(
            reverse_diffusion=self.reverse_ddpm,
            noise_predictor=self.noise_predictor,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            device='cpu'
        )

        # Test moving to CPU (should work)
        sampler_moved = sampler.to(torch.device('cpu'))
        self.assertEqual(sampler_moved.device, torch.device('cpu'))

    def test_error_conditions(self):
        """Test error conditions in sampling"""
        # Test conditional generation without conditional model
        sampler = SampleDDPM(
            reverse_diffusion=self.reverse_ddpm,
            noise_predictor=self.noise_predictor,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            device='cpu'
        )

        with self.assertRaises(ValueError):
            sampler(conditions=["test prompt"])

        # Test unconditional generation with conditional model
        sampler_conditional = SampleDDPM(
            reverse_diffusion=self.reverse_ddpm,
            noise_predictor=self.noise_predictor,
            image_shape=self.image_shape,
            conditional_model=self.conditional_model,
            batch_size=self.batch_size,
            device='cpu'
        )

        with self.assertRaises(ValueError):
            sampler_conditional(conditions=None)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete DDPM pipeline"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Create a simple dataset
        self.batch_size = 4
        self.channels = 1
        self.height = 16  # Smaller for faster tests
        self.width = 16

        x_data = torch.randn(16, self.channels, self.height, self.width)
        y_data = torch.arange(16)
        dataset = TensorDataset(x_data, y_data)

        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_complete_pipeline_unconditional(self):
        """Test complete pipeline: training -> sampling (unconditional)"""
        # Setup components
        variance_scheduler = VarianceSchedulerDDPM(num_steps=20, trainable_beta=False)  # Small for speed
        forward_ddpm = ForwardDDPM(variance_scheduler)
        reverse_ddpm = ReverseDDPM(variance_scheduler)
        noise_predictor = MockNoisePredictor(self.channels)
        optimizer = torch.optim.Adam(noise_predictor.parameters(), lr=1e-3)
        objective = nn.MSELoss()

        # Training
        trainer = TrainDDPM(
            noise_predictor=noise_predictor,
            forward_diffusion=forward_ddpm,
            reverse_diffusion=reverse_ddpm,
            data_loader=self.data_loader,
            optimizer=optimizer,
            objective=objective,
            max_epochs=1,  # Just 1 epoch for speed
            device='cpu',
            store_path=self.test_dir,
            log_frequency=1
        )

        train_losses, best_val_loss = trainer()

        # Sampling
        sampler = SampleDDPM(
            reverse_diffusion=reverse_ddpm,
            noise_predictor=noise_predictor,
            image_shape=(self.height, self.width),
            batch_size=2,
            in_channels=self.channels,
            device='cpu'
        )

        generated_imgs = sampler(save_images=False)

        # Verify results
        self.assertEqual(len(train_losses), 1)
        expected_shape = (2, self.channels, self.height, self.width)
        self.assertEqual(generated_imgs.shape, expected_shape)

    def test_complete_pipeline_conditional(self):
        """Test complete pipeline with conditional generation"""

        # Create string dataset for conditional training
        class StringDataset:
            def __init__(self, x_data, prompts):
                self.x_data = x_data
                self.prompts = prompts

            def __len__(self):
                return len(self.x_data)

            def __getitem__(self, idx):
                return self.x_data[idx], self.prompts[idx]

        x_data = torch.randn(8, self.channels, self.height, self.width)
        prompts = [f"image {i}" for i in range(8)]
        string_dataset = StringDataset(x_data, prompts)
        string_loader = DataLoader(string_dataset, batch_size=2, shuffle=True)

        # Setup components
        variance_scheduler = VarianceSchedulerDDPM(num_steps=20, trainable_beta=False)
        forward_ddpm = ForwardDDPM(variance_scheduler)
        reverse_ddpm = ReverseDDPM(variance_scheduler)
        noise_predictor = MockNoisePredictor(self.channels)
        conditional_model = MockConditionalModel()
        optimizer = torch.optim.Adam([
            *noise_predictor.parameters(),
            *conditional_model.parameters()
        ], lr=1e-3)
        objective = nn.MSELoss()

        # Training
        trainer = TrainDDPM(
            noise_predictor=noise_predictor,
            forward_diffusion=forward_ddpm,
            reverse_diffusion=reverse_ddpm,
            data_loader=string_loader,
            optimizer=optimizer,
            objective=objective,
            conditional_model=conditional_model,
            max_epochs=1,
            device='cpu',
            store_path=self.test_dir,
            log_frequency=1
        )

        train_losses, best_val_loss = trainer()

        # Conditional sampling
        sampler = SampleDDPM(
            reverse_diffusion=reverse_ddpm,
            noise_predictor=noise_predictor,
            image_shape=(self.height, self.width),
            conditional_model=conditional_model,
            batch_size=2,
            in_channels=self.channels,
            device='cpu'
        )

        test_prompts = ["test image 1", "test image 2"]
        generated_imgs = sampler(conditions=test_prompts, save_images=False)

        # Verify results
        self.assertEqual(len(train_losses), 1)
        expected_shape = (2, self.channels, self.height, self.width)
        self.assertEqual(generated_imgs.shape, expected_shape)

    def test_trainable_variance_schedule(self):
        """Test pipeline with trainable variance schedule"""
        variance_scheduler = VarianceSchedulerDDPM(num_steps=20, trainable_beta=True)
        forward_ddpm = ForwardDDPM(variance_scheduler)
        reverse_ddpm = ReverseDDPM(variance_scheduler)
        noise_predictor = MockNoisePredictor(self.channels)

        # Include variance scheduler parameters in optimizer
        all_params = list(noise_predictor.parameters()) + list(variance_scheduler.parameters())
        optimizer = torch.optim.Adam(all_params, lr=1e-3)
        objective = nn.MSELoss()

        trainer = TrainDDPM(
            noise_predictor=noise_predictor,
            forward_diffusion=forward_ddpm,
            reverse_diffusion=reverse_ddpm,
            data_loader=self.data_loader,
            optimizer=optimizer,
            objective=objective,
            max_epochs=1,
            device='cpu',
            store_path=self.test_dir,
            log_frequency=1
        )

        train_losses, best_val_loss = trainer()
        self.assertEqual(len(train_losses), 1)

    def test_checkpoint_save_load_cycle(self):
        """Test saving and loading checkpoints"""
        variance_scheduler = VarianceSchedulerDDPM(num_steps=20, trainable_beta=False)
        forward_ddpm = ForwardDDPM(variance_scheduler)
        reverse_ddpm = ReverseDDPM(variance_scheduler)
        noise_predictor = MockNoisePredictor(self.channels)
        optimizer = torch.optim.Adam(noise_predictor.parameters(), lr=1e-3)
        objective = nn.MSELoss()

        # Create first trainer and train
        trainer1 = TrainDDPM(
            noise_predictor=noise_predictor,
            forward_diffusion=forward_ddpm,
            reverse_diffusion=reverse_ddpm,
            data_loader=self.data_loader,
            optimizer=optimizer,
            objective=objective,
            max_epochs=1,
            device='cpu',
            store_path=self.test_dir,
            val_frequency=1
        )

        train_losses1, _ = trainer1()

        # Create new components and load checkpoint
        new_noise_predictor = MockNoisePredictor(self.channels)
        new_optimizer = torch.optim.Adam(new_noise_predictor.parameters(), lr=1e-3)

        trainer2 = TrainDDPM(
            noise_predictor=new_noise_predictor,
            forward_diffusion=forward_ddpm,
            reverse_diffusion=reverse_ddpm,
            data_loader=self.data_loader,
            optimizer=new_optimizer,
            objective=objective,
            max_epochs=1,
            device='cpu',
            store_path=self.test_dir
        )

        # Find checkpoint file
        checkpoint_files = [f for f in os.listdir(self.test_dir) if f.endswith('.pth')]
        self.assertTrue(len(checkpoint_files) > 0, "No checkpoint file found")

        checkpoint_path = os.path.join(self.test_dir, checkpoint_files[0])
        epoch, loss = trainer2.load_checkpoint(checkpoint_path)

        self.assertIsInstance(epoch, int)
        self.assertIsInstance(loss, float)

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_different_beta_schedules_consistency(self):
        """Test that different beta schedules produce valid results"""
        methods = ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]

        for method in methods:
            scheduler = VarianceSchedulerDDPM(
                num_steps=50,
                beta_method=method,
                trainable_beta=False
            )

            forward_ddpm = ForwardDDPM(scheduler)
            reverse_ddpm = ReverseDDPM(scheduler)

            # Test forward-reverse consistency
            x0 = torch.randn(2, 1, 16, 16)
            noise = torch.randn_like(x0)
            t = torch.randint(1, scheduler.num_steps - 1, (2,))

            # Forward process
            xt = forward_ddpm(x0, noise, t)

            # Reverse process (using ground truth noise)
            xt_minus_1 = reverse_ddpm(xt, noise, t)

            # Results should be finite
            self.assertTrue(torch.all(torch.isfinite(xt)))
            self.assertTrue(torch.all(torch.isfinite(xt_minus_1)))

    def test_extreme_timesteps(self):
        """Test behavior at extreme timesteps"""
        scheduler = VarianceSchedulerDDPM(num_steps=100, trainable_beta=False)
        forward_ddpm = ForwardDDPM(scheduler)
        reverse_ddpm = ReverseDDPM(scheduler)

        x0 = torch.randn(2, 1, 16, 16)
        noise = torch.randn_like(x0)

        # Test t=0 (minimal noise)
        t_zero = torch.zeros(2, dtype=torch.long)
        xt_zero = forward_ddpm(x0, noise, t_zero)

        # At t=0, result should be very close to original
        alpha_bar_0 = scheduler.alpha_bars[0]
        expected_xt_zero = torch.sqrt(alpha_bar_0) * x0 + torch.sqrt(1 - alpha_bar_0) * noise
        torch.testing.assert_close(xt_zero, expected_xt_zero, atol=1e-6, rtol=1e-6)

        # Test t=max-1 (maximum noise)
        t_max = torch.full((2,), scheduler.num_steps - 1, dtype=torch.long)
        xt_max = forward_ddpm(x0, noise, t_max)

        # Result should be mostly noise
        self.assertTrue(torch.all(torch.isfinite(xt_max)))

    def test_batch_size_consistency(self):
        """Test consistency across different batch sizes"""
        scheduler = VarianceSchedulerDDPM(num_steps=50, trainable_beta=False)
        forward_ddpm = ForwardDDPM(scheduler)

        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            x0 = torch.randn(batch_size, 3, 16, 16)
            noise = torch.randn_like(x0)
            t = torch.randint(0, scheduler.num_steps, (batch_size,))

            xt = forward_ddpm(x0, noise, t)

            self.assertEqual(xt.shape[0], batch_size)
            self.assertTrue(torch.all(torch.isfinite(xt)))

    def test_memory_efficiency(self):
        """Test that operations don't cause memory leaks (basic check)"""
        scheduler = VarianceSchedulerDDPM(num_steps=100, trainable_beta=False)
        forward_ddpm = ForwardDDPM(scheduler)

        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Perform multiple operations
        for _ in range(10):
            x0 = torch.randn(4, 3, 32, 32)
            noise = torch.randn_like(x0)
            t = torch.randint(0, scheduler.num_steps, (4,))

            xt = forward_ddpm(x0, noise, t)
            del xt, x0, noise, t

        # Memory usage shouldn't grow significantly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            # Allow for some variance in memory usage
            self.assertLess(final_memory - initial_memory, 100 * 1024 * 1024)  # 100MB threshold

def run_performance_benchmarks():
    """Optional performance benchmarks (not part of main test suite)"""
    print("\n" + "=" * 50)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 50)

    import time

    # Benchmark forward diffusion
    scheduler = VarianceSchedulerDDPM(num_steps=1000, trainable_beta=False)
    forward_ddpm = ForwardDDPM(scheduler)

    x0 = torch.randn(32, 3, 64, 64)
    noise = torch.randn_like(x0)
    t = torch.randint(0, scheduler.num_steps, (32,))

    # Warmup
    for _ in range(5):
        _ = forward_ddpm(x0, noise, t)

    # Benchmark
    start_time = time.time()
    for _ in range(100):
        xt = forward_ddpm(x0, noise, t)
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    print(f"Forward diffusion avg time: {avg_time * 1000:.2f}ms per batch")
    print(f"Throughput: {32 / avg_time:.0f} images/second")

class DDPMTestSuite:
    """Main test suite runner with custom reporting"""

    def __init__(self):
        self.test_classes = [
            TestVarianceSchedulerDDPM,
            TestForwardDDPM,
            TestReverseDDPM,
            TestTrainDDPM,
            TestSampleDDPM,
            TestIntegration,
            TestEdgeCases
        ]

    def run_all_tests(self, verbose=True):
        """Run all tests with custom reporting"""
        print("=" * 60)
        print("DDPM IMPLEMENTATION TEST SUITE")
        print("=" * 60)

        total_tests = 0
        total_failures = 0
        total_errors = 0

        for test_class in self.test_classes:
            print(f"\nRunning {test_class.__name__}...")

            # Create test suite for this class
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)

            # Run tests with custom result handling
            result = unittest.TextTestRunner(
                verbosity=2 if verbose else 1,
                stream=open(os.devnull, 'w') if not verbose else None
            ).run(suite)

            # Count results
            tests_run = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)

            total_tests += tests_run
            total_failures += failures
            total_errors += errors

            # Print summary for this class
            status = "PASS" if (failures == 0 and errors == 0) else "FAIL"
            print(f"  {test_class.__name__}: {status}")
            print(f"    Tests: {tests_run}, Failures: {failures}, Errors: {errors}")

            # Print failure details
            if failures > 0:
                print("    FAILURES:")
                for test, traceback in result.failures:
                    print(
                        f"      - {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'Unknown failure'}")

            if errors > 0:
                print("    ERRORS:")
                for test, traceback in result.errors:
                    error_msg = traceback.split('\n')[-2] if len(traceback.split('\n')) > 1 else "Unknown error"
                    print(f"      - {test}: {error_msg}")

        # Final summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_tests - total_failures - total_errors}")
        print(f"Failed: {total_failures}")
        print(f"Errors: {total_errors}")

        success_rate = ((total_tests - total_failures - total_errors) / total_tests) * 100 if total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")

        if total_failures == 0 and total_errors == 0:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"\n❌ {total_failures + total_errors} tests failed/errored")

        return total_failures == 0 and total_errors == 0

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Run main test suite
    suite = DDPMTestSuite()
    all_passed = suite.run_all_tests(verbose=True)

    # Optionally run performance benchmarks
    run_benchmarks = input("\nRun performance benchmarks? (y/n): ").lower().startswith('y')
    if run_benchmarks:
        run_performance_benchmarks()

    # Exit with appropriate code
    exit(0 if all_passed else 1)