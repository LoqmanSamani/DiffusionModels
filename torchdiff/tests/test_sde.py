import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch
from torch.utils.data import DataLoader, TensorDataset
from torchdiff.utils import NoisePredictor, TextEncoder
from torchdiff.sde import (
    VarianceSchedulerSDE,
    ForwardSDE,
    ReverseSDE,
    TrainSDE,
    SampleSDE
)


class TestVarianceSchedulerSDE:
    """Test cases for VarianceSchedulerSDE class."""

    def test_init(self):
        """Test initialization with different parameters."""
        # Test default initialization
        scheduler = VarianceSchedulerSDE()
        assert scheduler.num_steps == 1000
        assert scheduler.beta_start == 1e-4
        assert scheduler.beta_end == 0.02

        # Test custom initialization
        scheduler = VarianceSchedulerSDE(
            num_steps=500,
            beta_start=1e-5,
            beta_end=0.01,
            beta_method="quadratic"
        )
        assert scheduler.num_steps == 500
        assert scheduler.beta_start == 1e-5
        assert scheduler.beta_end == 0.01

    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            VarianceSchedulerSDE(num_steps=-10)

        with pytest.raises(ValueError):
            VarianceSchedulerSDE(beta_start=0.1, beta_end=0.01)  # start > end

        with pytest.raises(ValueError):
            VarianceSchedulerSDE(sigma_start=1.0, sigma_end=0.5)  # start > end

    def test_beta_schedule_methods(self):
        """Test all beta schedule computation methods."""
        methods = ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]

        for method in methods:
            scheduler = VarianceSchedulerSDE(num_steps=100, beta_method=method)
            betas = scheduler.betas

            assert betas.shape[0] == 100
            assert torch.all(betas >= scheduler.beta_start)
            assert torch.all(betas <= scheduler.beta_end)

    def test_cumulative_betas(self):
        """Test cumulative beta computation."""
        scheduler = VarianceSchedulerSDE(num_steps=100)
        cum_betas = scheduler._cum_betas

        assert cum_betas.shape[0] == 100
        assert cum_betas[0] > 0  # Should be positive
        assert cum_betas[-1] > cum_betas[0]  # Should be increasing

    def test_sigmas(self):
        """Test sigma computation."""
        scheduler = VarianceSchedulerSDE(num_steps=100)
        sigmas = scheduler.sigmas

        assert sigmas.shape[0] == 100
        assert sigmas[0] == scheduler.sigma_start
        assert sigmas[-1] == scheduler.sigma_end

    def test_get_variance(self):
        """Test variance computation for different methods."""
        scheduler = VarianceSchedulerSDE(num_steps=100)
        time_steps = torch.tensor([0, 50, 99])

        for method in ["ve", "vp", "sub-vp"]:
            variance = scheduler.get_variance(time_steps, method)
            assert variance.shape[0] == 3
            assert torch.all(variance >= 0)  # Variance should be non-negative


class TestForwardSDE:
    """Test cases for ForwardSDE class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.scheduler = VarianceSchedulerSDE(num_steps=100)
        self.batch_size = 4
        self.channels = 3
        self.height = 32
        self.width = 32

    def test_init(self):
        """Test initialization."""
        for method in ["ve", "vp", "sub-vp", "ode"]:
            forward_sde = ForwardSDE(self.scheduler, method)
            assert forward_sde.sde_method == method

        with pytest.raises(ValueError):
            ForwardSDE(self.scheduler, "invalid_method")

    def test_forward_ve(self):
        """Test VE forward process."""
        forward_sde = ForwardSDE(self.scheduler, "ve")
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt = forward_sde(x0, noise, time_steps)
        assert xt.shape == x0.shape

    def test_forward_vp(self):
        """Test VP forward process."""
        forward_sde = ForwardSDE(self.scheduler, "vp")
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt = forward_sde(x0, noise, time_steps)
        assert xt.shape == x0.shape

    def test_forward_sub_vp(self):
        """Test sub-VP forward process."""
        forward_sde = ForwardSDE(self.scheduler, "sub-vp")
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt = forward_sde(x0, noise, time_steps)
        assert xt.shape == x0.shape

    def test_forward_ode(self):
        """Test ODE forward process."""
        forward_sde = ForwardSDE(self.scheduler, "ode")
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(x0)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt = forward_sde(x0, noise, time_steps)
        assert xt.shape == x0.shape


class TestReverseSDE:
    """Test cases for ReverseSDE class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.scheduler = VarianceSchedulerSDE(num_steps=100)
        self.batch_size = 4
        self.channels = 3
        self.height = 32
        self.width = 32

    def test_init(self):
        """Test initialization."""
        for method in ["ve", "vp", "sub-vp", "ode"]:
            reverse_sde = ReverseSDE(self.scheduler, method)
            assert reverse_sde.sde_method == method

        with pytest.raises(ValueError):
            ReverseSDE(self.scheduler, "invalid_method")

    def test_reverse_ve(self):
        """Test VE reverse process."""
        reverse_sde = ReverseSDE(self.scheduler, "ve")
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(xt)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(1, 100, (self.batch_size,))

        xt_prev = reverse_sde(xt, noise, predicted_noise, time_steps)
        assert xt_prev.shape == xt.shape

    def test_reverse_vp(self):
        """Test VP reverse process."""
        reverse_sde = ReverseSDE(self.scheduler, "vp")
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(xt)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt_prev = reverse_sde(xt, noise, predicted_noise, time_steps)
        assert xt_prev.shape == xt.shape

    def test_reverse_sub_vp(self):
        """Test sub-VP reverse process."""
        reverse_sde = ReverseSDE(self.scheduler, "sub-vp")
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = torch.randn_like(xt)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt_prev = reverse_sde(xt, noise, predicted_noise, time_steps)
        assert xt_prev.shape == xt.shape

    def test_reverse_ode(self):
        """Test ODE reverse process."""
        reverse_sde = ReverseSDE(self.scheduler, "ode")
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        noise = None  # ODE doesn't use noise
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.randint(0, 100, (self.batch_size,))

        xt_prev = reverse_sde(xt, noise, predicted_noise, time_steps)
        assert xt_prev.shape == xt.shape


class TestTrainSDE:
    """Test cases for TrainSDE class."""

    def setup_method(self):
        """Setup test fixtures."""

        # Create simple models for testing
        class SimpleNoisePredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)

            def forward(self, x, t, y=None, mask=None):
                return self.conv(x)

        class SimpleConditionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(77, 64)

            def forward(self, input_ids, attention_mask=None):
                return self.embed(input_ids.float())

        # Create test data
        self.batch_size = 4
        self.channels = 3
        self.height = 32
        self.width = 32

        x_data = torch.randn(20, self.channels, self.height, self.width)
        y_data = torch.randint(0, 10, (20,))
        dataset = TensorDataset(x_data, y_data)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size)

        # Create components
        self.scheduler = VarianceSchedulerSDE(num_steps=10)
        self.forward_sde = ForwardSDE(self.scheduler, "vp")
        self.reverse_sde = ReverseSDE(self.scheduler, "vp")
        self.noise_predictor = SimpleNoisePredictor()
        self.conditional_model = SimpleConditionalModel()
        self.optimizer = torch.optim.Adam(
            list(self.noise_predictor.parameters()) +
            list(self.conditional_model.parameters()),
            lr=1e-4
        )
        self.objective = nn.MSELoss()

    def test_init(self):
        """Test initialization."""
        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            conditional_model=self.conditional_model,
            max_epochs=2
        )

        assert trainer is not None


    @patch('sde.TrainSDE._setup_ddp')
    def test_ddp_setup(self, mock_setup_ddp):
        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            use_ddp=True
        )

        mock_setup_ddp.assert_called_once()


    def test_single_gpu_setup(self):
        """Test single GPU setup."""
        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            use_ddp=False
        )

        assert trainer.ddp_rank == 0
        assert trainer.ddp_local_rank == 0
        assert trainer.ddp_world_size == 1
        assert trainer.master_process

    def test_warmup_scheduler(self):
        """Test warmup scheduler creation."""
        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective
        )

        scheduler = trainer.warmup_scheduler(self.optimizer, 10)
        assert scheduler is not None

    def test_process_conditional_input(self):
        """Test conditional input processing."""
        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            conditional_model=self.conditional_model
        )

        # Test with tensor input
        y_tensor = torch.tensor([1, 2, 3, 4])
        y_encoded = trainer._process_conditional_input(y_tensor)
        assert y_encoded is not None

        # Test with list input
        y_list = ["test1", "test2", "test3", "test4"]
        y_encoded = trainer._process_conditional_input(y_list)
        assert y_encoded is not None

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = TrainSDE(
                noise_predictor=self.noise_predictor,
                forward_diffusion=self.forward_sde,
                reverse_diffusion=self.reverse_sde,
                data_loader=self.data_loader,
                optimizer=self.optimizer,
                objective=self.objective,
                store_path=temp_dir
            )

            trainer._save_checkpoint(1, 0.5)

            # Check if file was created
            files = os.listdir(temp_dir)
            assert any(f.startswith("sde_epoch_1") for f in files)

    def test_validate(self):
        """Test validation method."""
        # Mock metrics
        mock_metrics = Mock()
        mock_metrics.forward.return_value = (1.0, 0.1, 25.0, 0.8, 0.2)
        mock_metrics.fid = True
        mock_metrics.metrics = True
        mock_metrics.lpips = True

        trainer = TrainSDE(
            noise_predictor=self.noise_predictor,
            forward_diffusion=self.forward_sde,
            reverse_diffusion=self.reverse_sde,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
            objective=self.objective,
            val_loader=self.data_loader,
            metrics_=mock_metrics
        )

        val_loss, fid, mse, psnr, ssim, lpips = trainer.validate()

        assert isinstance(val_loss, float)
        assert isinstance(fid, float)
        assert isinstance(mse, float)
        assert isinstance(psnr, float)
        assert isinstance(ssim, float)
        assert isinstance(lpips, float)


class TestSampleSDE:
    """Test cases for SampleSDE class."""

    def setup_method(self):
        """Setup test fixtures."""

        # Create simple models for testing
        class SimpleNoisePredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)

            def forward(self, x, t, y=None):
                return self.conv(x)

        class SimpleConditionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(77, 64)

            def forward(self, input_ids, attention_mask=None):
                return self.embed(input_ids.float())

        # Create components
        self.scheduler = VarianceSchedulerSDE(num_steps=10)
        self.reverse_sde = ReverseSDE(self.scheduler, "vp")
        self.noise_predictor = SimpleNoisePredictor()
        self.conditional_model = SimpleConditionalModel()

    def test_init(self):
        """Test initialization."""
        sampler = SampleSDE(
            reverse_diffusion=self.reverse_sde,
            noise_predictor=self.noise_predictor,
            image_shape=(32, 32)
        )

        assert sampler is not None

    def test_tokenize(self):
        """Test tokenization method."""
        sampler = SampleSDE(
            reverse_diffusion=self.reverse_sde,
            noise_predictor=self.noise_predictor,
            image_shape=(32, 32),
            conditional_model=self.conditional_model
        )

        # Test with single prompt
        input_ids, attention_mask = sampler.tokenize("a test prompt")
        assert input_ids.shape[0] == 1
        assert attention_mask.shape[0] == 1

        # Test with multiple prompts
        input_ids, attention_mask = sampler.tokenize(["prompt1", "prompt2"])
        assert input_ids.shape[0] == 2
        assert attention_mask.shape[0] == 2

    def test_forward_unconditional(self):
        """Test unconditional sampling."""
        sampler = SampleSDE(
            reverse_diffusion=self.reverse_sde,
            noise_predictor=self.noise_predictor,
            image_shape=(32, 32),
            batch_size=2
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            images = sampler.forward(
                conditions=None,
                save_images=True,
                save_path=temp_dir
            )

            assert images.shape == (2, 3, 32, 32)
            assert torch.all(images >= 0) and torch.all(images <= 1)  # Normalized

            # Check if images were saved
            files = os.listdir(temp_dir)
            assert len(files) == 2

    def test_forward_conditional(self):
        """Test conditional sampling."""
        sampler = SampleSDE(
            reverse_diffusion=self.reverse_sde,
            noise_predictor=self.noise_predictor,
            image_shape=(32, 32),
            conditional_model=self.conditional_model,
            batch_size=2
        )

        images = sampler.forward(
            conditions=["a cat", "a dog"],
            save_images=False
        )

        assert images.shape == (2, 3, 32, 32)

    def test_to_device(self):
        """Test device movement."""
        sampler = SampleSDE(
            reverse_diffusion=self.reverse_sde,
            noise_predictor=self.noise_predictor,
            image_shape=(32, 32)
        )

        # Move to CPU if CUDA is available, otherwise test stays on CPU
        target_device = torch.device("cpu")
        sampler = sampler.to(target_device)

        assert sampler.device == target_device
        assert next(sampler.noise_predictor.parameters()).device == target_device
        assert next(sampler.reverse.parameters()).device == target_device


def test_integration():
    """Integration test with the provided usage code."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create simple test data
    x_data = torch.randn(20, 3, 32, 32)
    y_data = torch.randint(0, 10, (20,))
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # Initialize models with smaller parameters for testing
    noise_predictor = NoisePredictor(
        in_channels=3,
        down_channels=[8, 16],  # Reduced channels for testing
        mid_channels=[16, 16],
        up_channels=[16, 8],
        down_sampling=[True, False],  # Only one downsampling for small images
        time_embed_dim=32,
        y_embed_dim=32,
        num_down_blocks=1,
        num_mid_blocks=1,
        num_up_blocks=1,
        down_sampling_factor=2
    ).to(device)

    text_encoder = TextEncoder(
        use_pretrained_model=False,  # Don't use pretrained for faster testing
        model_name="bert-base-uncased",
        vocabulary_size=100,  # Smaller vocabulary
        num_layers=1,  # Fewer layers
        input_dimension=32,
        output_dimension=32,
        num_heads=2,
        context_length=10  # Shorter context
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        [p for p in noise_predictor.parameters() if p.requires_grad] +
        [p for p in text_encoder.parameters() if p.requires_grad],
        lr=1e-4
    )
    loss = nn.MSELoss()

    # SDE hyperparameters with fewer steps
    hyperparams_sde = VarianceSchedulerSDE(
        num_steps=10,  # Fewer steps for testing
        beta_start=1e-4,
        beta_end=0.02,
        trainable_beta=False,
        sigma_start=1e-3,
        sigma_end=10.0,
        start=0.0,
        end=1.0,
        beta_method="linear"
    )

    # Forward and reverse SDE
    forward_sde = ForwardSDE(variance_scheduler=hyperparams_sde, sde_method="vp")
    reverse_sde = ReverseSDE(variance_scheduler=hyperparams_sde, sde_method="vp")

    # TrainSDE with minimal settings
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = TrainSDE(
            noise_predictor=noise_predictor,
            forward_diffusion=forward_sde,
            reverse_diffusion=reverse_sde,
            data_loader=train_loader,
            optimizer=optimizer,
            objective=loss,
            val_loader=val_loader,
            max_epochs=2,  # Just 2 epochs for testing
            device=device,
            conditional_model=text_encoder,
            metrics_=None,  # No metrics for faster testing
            store_path=temp_dir,
            val_frequency=1,
            use_ddp=False,
            grad_accumulation_steps=1,
            log_frequency=1,
            use_compilation=False
        )

        # Test training
        train_losses, best_val_loss = trainer()
        assert len(train_losses) >= 0  # Could be empty if early stopping
        assert isinstance(best_val_loss, float)

        # Test sampling
        sampler = SampleSDE(
            reverse_diffusion=reverse_sde,
            noise_predictor=noise_predictor,
            image_shape=(32, 32),
            conditional_model=text_encoder,
            tokenizer="bert-base-uncased",
            max_token_length=10,  # Shorter for testing
            batch_size=2,
            in_channels=3,
            device=device,
            image_output_range=(-1.0, 1.0)
        )

        # Test with class names
        class_names = ['airplane', 'automobile']
        images = sampler(class_names, save_images=False)
        assert images.shape == (2, 3, 32, 32)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])