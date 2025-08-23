import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np

# Import the LDM components (assuming they're in ldm.py)
from torchdiff.ldm import (
    AutoencoderLDM, TrainAE, TrainLDM, SampleLDM,
    VectorQuantizer, DownBlock, UpBlock, Conv3,
    DownSampling, UpSampling, Attention
)
from torchdiff.sde import ForwardSDE, ReverseSDE, VarianceSchedulerSDE



# mock utility classes that would normally come from torchdiff.utils.py
class MockTextEncoder(nn.Module):
    def __init__(self, output_dim=32):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        return torch.randn(batch_size, self.output_dim)


class MockNoisePredictor(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x, t, y=None, context=None):
        return self.conv(x)


class MockMetrics:
    def __init__(self, device="cpu", fid=True, metrics=True, lpips_=True):
        self.device = device
        self.fid = fid
        self.metrics = metrics
        self.lpips = lpips_

    def forward(self, x_real, x_fake):
        return (
            torch.tensor(10.0) if self.fid else float('inf'),  # FID
            torch.tensor(0.1) if self.metrics else None,  # MSE
            torch.tensor(25.0) if self.metrics else None,  # PSNR
            torch.tensor(0.8) if self.metrics else None,  # SSIM
            torch.tensor(0.2) if self.lpips else None  # LPIPS
        )


class TestAutoencoderLDM:
    """Test suite for AutoencoderLDM component."""

    @pytest.fixture
    def autoencoder_config(self):
        """Standard configuration for autoencoder tests."""
        return {
            'in_channels': 3,
            'down_channels': [8, 16],
            'up_channels': [16, 8],
            'out_channels': 3,
            'dropout_rate': 0.1,
            'num_heads': 1,
            'num_groups': 8,
            'num_layers_per_block': 2,
            'total_down_sampling_factor': 2,
            'latent_channels': 4,
            'num_embeddings': 32,
            'use_vq': False,
            'beta': 1.0
        }

    @pytest.fixture
    def sample_data(self):
        """Sample input data for testing."""
        return torch.randn(2, 3, 32, 32)

    def test_autoencoder_initialization(self, autoencoder_config):
        """Test AutoencoderLDM initialization with different configurations."""
        # Test KL-divergence mode
        model_kl = AutoencoderLDM(**autoencoder_config)
        assert not model_kl.use_vq
        assert model_kl.beta == 1.0

        # Test VQ mode
        config_vq = autoencoder_config.copy()
        config_vq['use_vq'] = True
        model_vq = AutoencoderLDM(**config_vq)
        assert model_vq.use_vq
        assert hasattr(model_vq, 'vq_layer')

    def test_autoencoder_forward_pass(self, autoencoder_config, sample_data):
        """Test forward pass of AutoencoderLDM."""
        model = AutoencoderLDM(**autoencoder_config)
        model.eval()

        with torch.no_grad():
            x_hat, total_loss, reg_loss, z = model(sample_data)

        # Check output shapes
        assert x_hat.shape == sample_data.shape
        assert isinstance(total_loss, float)
        assert isinstance(reg_loss, (float, torch.Tensor))

        # Check latent shape
        expected_latent_shape = (2, 4, 16, 16)  # Based on config
        assert z.shape == expected_latent_shape

    def test_encode_decode_consistency(self, autoencoder_config, sample_data):
        """Test encode-decode cycle."""
        model = AutoencoderLDM(**autoencoder_config)
        model.eval()

        with torch.no_grad():
            z, reg_loss = model.encode(sample_data)
            x_reconstructed = model.decode(z)

        assert x_reconstructed.shape == sample_data.shape
        assert z.shape[0] == sample_data.shape[0]  # Batch size preserved

    def test_vq_functionality(self, autoencoder_config, sample_data):
        """Test Vector Quantization functionality."""
        config_vq = autoencoder_config.copy()
        config_vq['use_vq'] = True
        model = AutoencoderLDM(**config_vq)
        model.eval()

        with torch.no_grad():
            z, vq_loss = model.encode(sample_data)

        assert isinstance(vq_loss, torch.Tensor)
        assert vq_loss.requires_grad == False

    def test_reparameterization_trick(self, autoencoder_config):
        """Test reparameterization trick for VAE."""
        model = AutoencoderLDM(**autoencoder_config)

        mu = torch.randn(2, 8, 16, 16)
        logvar = torch.randn(2, 8, 16, 16)

        z = model.reparameterize(mu, logvar)
        assert z.shape == mu.shape

        # Test deterministic behavior when logvar is very negative
        logvar_zero = torch.full_like(logvar, -20)
        z_det = model.reparameterize(mu, logvar_zero)
        torch.testing.assert_close(z_det, mu, atol=1e-3, rtol=1e-3)


class TestVectorQuantizer:
    """Test suite for VectorQuantizer component."""

    def test_vq_initialization(self):
        """Test VectorQuantizer initialization."""
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=32)

        assert vq.num_embeddings == 64
        assert vq.embedding_dim == 32
        assert vq.embedding.weight.shape == (64, 32)

    def test_vq_forward_pass(self):
        """Test VectorQuantizer forward pass."""
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=16)
        x = torch.randn(2, 16, 8, 8)

        quantized, vq_loss = vq(x)

        assert quantized.shape == x.shape
        assert isinstance(vq_loss, torch.Tensor)
        assert vq_loss.requires_grad


class TestConvolutionalBlocks:
    """Test suite for convolutional building blocks."""

    def test_conv3_block(self):
        """Test Conv3 block."""
        conv3 = Conv3(in_channels=16, out_channels=32, dropout_rate=0.1)
        x = torch.randn(2, 16, 32, 32)

        output = conv3(x)
        assert output.shape == (2, 32, 32, 32)

    def test_down_block(self):
        """Test DownBlock."""
        down_block = DownBlock(
            in_channels=16, out_channels=32,
            num_layers=2, down_sampling_factor=2,
            dropout_rate=0.1
        )
        x = torch.randn(2, 16, 32, 32)

        output = down_block(x)
        assert output.shape == (2, 32, 16, 16)  # Downsampled by factor of 2

    def test_up_block(self):
        """Test UpBlock."""
        up_block = UpBlock(
            in_channels=32, out_channels=16,
            num_layers=2, up_sampling_factor=2,
            dropout_rate=0.1
        )
        x = torch.randn(2, 32, 16, 16)

        output = up_block(x)
        assert output.shape == (2, 16, 32, 32)  # Upsampled by factor of 2

    def test_attention_block(self):
        """Test Attention block."""
        attention = Attention(
            num_channels=32, num_heads=4,
            num_groups=8, dropout_rate=0.1
        )
        x = torch.randn(2, 32, 16, 16)

        output = attention(x)
        assert output.shape == x.shape


class TestSamplingLayers:
    """Test suite for sampling layers."""

    def test_down_sampling(self):
        """Test DownSampling layer."""
        down_sample = DownSampling(
            in_channels=16, out_channels=32,
            down_sampling_factor=2
        )
        x = torch.randn(2, 16, 32, 32)

        output = down_sample(x)
        assert output.shape == (2, 32, 16, 16)

    def test_up_sampling(self):
        """Test UpSampling layer."""
        up_sample = UpSampling(
            in_channels=32, out_channels=16,
            up_sampling_factor=2
        )
        x = torch.randn(2, 32, 16, 16)

        output = up_sample(x)
        assert output.shape == (2, 16, 32, 32)


class TestVarianceSchedulerSDE:
    """Test suite for SDE variance scheduler."""

    def test_scheduler_initialization(self):
        """Test VarianceSchedulerSDE initialization."""
        scheduler = VarianceSchedulerSDE(
            num_steps=100, beta_start=1e-4, beta_end=0.02,
            trainable_beta=False
        )

        assert scheduler.num_steps == 100
        assert scheduler.beta_start == 1e-4
        assert scheduler.beta_end == 0.02
        assert not scheduler.trainable_beta

    def test_beta_schedules(self):
        """Test different beta scheduling methods."""
        methods = ["linear", "sigmoid", "quadratic", "constant"]

        for method in methods:
            scheduler = VarianceSchedulerSDE(
                num_steps=100, beta_start=1e-4, beta_end=0.02,
                beta_method=method, trainable_beta=False
            )

            betas = scheduler.betas
            assert betas.shape == (100,)
            assert torch.all(betas >= 1e-4)
            assert torch.all(betas <= 0.02)

    def test_trainable_beta(self):
        """Test trainable beta functionality."""
        scheduler = VarianceSchedulerSDE(
            num_steps=50, trainable_beta=True
        )

        # Check that beta_raw is a parameter
        assert hasattr(scheduler, 'beta_raw')
        assert isinstance(scheduler.beta_raw, nn.Parameter)

        # Check that betas are in valid range
        betas = scheduler.betas
        assert torch.all(betas >= scheduler.beta_start)
        assert torch.all(betas <= scheduler.beta_end)

    def test_variance_computation(self):
        """Test variance computation for different SDE methods."""
        scheduler = VarianceSchedulerSDE(num_steps=100, trainable_beta=False)
        time_steps = torch.tensor([10, 20, 30])

        methods = ["ve", "vp", "sub-vp"]
        for method in methods:
            variance = scheduler.get_variance(time_steps, method)
            assert variance.shape == time_steps.shape
            assert torch.all(variance >= 0)


class TestSDEProcesses:
    """Test suite for SDE forward and reverse processes."""

    @pytest.fixture
    def sde_setup(self):
        """Setup SDE components for testing."""
        scheduler = VarianceSchedulerSDE(
            num_steps=100, beta_start=1e-4, beta_end=0.02
        )
        return scheduler

    def test_forward_sde_methods(self, sde_setup):
        """Test ForwardSDE with different methods."""
        methods = ["ve", "vp", "sub-vp", "ode"]
        x0 = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(x0)
        time_steps = torch.tensor([10, 20])

        for method in methods:
            forward_sde = ForwardSDE(sde_setup, method)
            xt = forward_sde(x0, noise, time_steps)

            assert xt.shape == x0.shape
            assert not torch.isnan(xt).any()

    def test_reverse_sde_methods(self, sde_setup):
        """Test ReverseSDE with different methods."""
        methods = ["ve", "vp", "sub-vp", "ode"]
        xt = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(xt)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.tensor([10, 20])

        for method in methods:
            reverse_sde = ReverseSDE(sde_setup, method)
            x_prev = reverse_sde(xt, noise, predicted_noise, time_steps)

            assert x_prev.shape == xt.shape
            assert not torch.isnan(x_prev).any()

    def test_ode_method_without_noise(self, sde_setup):
        """Test ODE method works without noise."""
        forward_sde = ForwardSDE(sde_setup, "ode")
        reverse_sde = ReverseSDE(sde_setup, "ode")

        x0 = torch.randn(2, 3, 32, 32)
        xt = torch.randn_like(x0)
        predicted_noise = torch.randn_like(x0)
        time_steps = torch.tensor([10, 20])

        # Forward SDE with ODE method
        xt_forward = forward_sde(x0, torch.randn_like(x0), time_steps)

        # Reverse SDE with ODE method (noise can be None)
        x_prev = reverse_sde(xt, None, predicted_noise, time_steps)

        assert not torch.isnan(xt_forward).any()
        assert not torch.isnan(x_prev).any()


class TestTrainAE:
    """Test suite for AutoEncoder trainer."""

    @pytest.fixture
    def training_setup(self):
        """Setup training components."""
        # Create simple dataset
        data = torch.randn(20, 3, 32, 32)
        labels = torch.randint(0, 10, (20,))
        dataset = TensorDataset(data, labels)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        # Create model and optimizer
        model = AutoencoderLDM(
            in_channels=3, down_channels=[8, 16], up_channels=[16, 8],
            out_channels=3, dropout_rate=0.1, latent_channels=4,
            num_heads=1, num_groups=8, num_layers_per_block=2,
            total_down_sampling_factor=2, num_embeddings=32
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = MockMetrics()

        return {
            'model': model,
            'optimizer': optimizer,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'metrics': metrics
        }

    def test_train_ae_initialization(self, training_setup):
        """Test TrainAE initialization."""
        trainer = TrainAE(
            model=training_setup['model'],
            optimizer=training_setup['optimizer'],
            data_loader=training_setup['train_loader'],
            val_loader=training_setup['val_loader'],
            max_epochs=2,
            metrics_=training_setup['metrics'],
            device='cpu'
        )

        assert trainer.max_epochs == 2
        assert trainer.device.type == 'cpu'

    def test_train_ae_forward(self, training_setup):
        """Test TrainAE training loop."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = TrainAE(
                model=training_setup['model'],
                optimizer=training_setup['optimizer'],
                data_loader=training_setup['train_loader'],
                val_loader=training_setup['val_loader'],
                max_epochs=2,
                metrics_=training_setup['metrics'],
                device='cpu',
                store_path=temp_dir,
                val_frequency=1,
                log_frequency=1
            )

            train_losses, best_val_loss = trainer()

            assert len(train_losses) <= 2
            assert isinstance(best_val_loss, float)
            assert best_val_loss < float('inf')


class TestTrainLDM:
    """Test suite for LDM trainer."""

    @pytest.fixture
    def ldm_training_setup(self):
        """Setup LDM training components."""
        # Create dataset with text labels
        data = torch.randn(16, 3, 32, 32)
        labels = ['class_' + str(i % 4) for i in range(16)]
        dataset = TensorDataset(data, labels)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        # Create components
        compressor = AutoencoderLDM(
            in_channels=3, down_channels=[8, 16], up_channels=[16, 8],
            out_channels=3, dropout_rate=0.1, latent_channels=4,
            num_heads=1, num_groups=8, num_layers_per_block=2,
            total_down_sampling_factor=2, num_embeddings=32
        )

        noise_predictor = MockNoisePredictor(in_channels=4)
        text_encoder = MockTextEncoder(output_dim=32)

        scheduler = VarianceSchedulerSDE(num_steps=50, trainable_beta=False)
        forward_sde = ForwardSDE(scheduler, "ode")
        reverse_sde = ReverseSDE(scheduler, "ode")

        optimizer = torch.optim.Adam([
            *noise_predictor.parameters(),
            *text_encoder.parameters()
        ], lr=1e-3)

        return {
            'compressor': compressor,
            'noise_predictor': noise_predictor,
            'text_encoder': text_encoder,
            'forward_sde': forward_sde,
            'reverse_sde': reverse_sde,
            'optimizer': optimizer,
            'train_loader': train_loader,
            'val_loader': val_loader
        }

    def test_train_ldm_initialization(self, ldm_training_setup):
        """Test TrainLDM initialization."""
        setup = ldm_training_setup

        trainer = TrainLDM(
            diffusion_model="sde",
            forward_diffusion=setup['forward_sde'],
            reverse_diffusion=setup['reverse_sde'],
            noise_predictor=setup['noise_predictor'],
            compressor_model=setup['compressor'],
            optimizer=setup['optimizer'],
            objective=nn.MSELoss(),
            data_loader=setup['train_loader'],
            val_loader=setup['val_loader'],
            conditional_model=setup['text_encoder'],
            max_epochs=2,
            device='cpu'
        )

        assert trainer.diffusion_model == "sde"
        assert trainer.max_epochs == 2

    def test_train_ldm_forward(self, ldm_training_setup):
        """Test TrainLDM training loop."""
        setup = ldm_training_setup

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = TrainLDM(
                diffusion_model="sde",
                forward_diffusion=setup['forward_sde'],
                reverse_diffusion=setup['reverse_sde'],
                noise_predictor=setup['noise_predictor'],
                compressor_model=setup['compressor'],
                optimizer=setup['optimizer'],
                objective=nn.MSELoss(),
                data_loader=setup['train_loader'],
                val_loader=setup['val_loader'],
                conditional_model=setup['text_encoder'],
                max_epochs=2,
                device='cpu',
                store_path=temp_dir,
                val_frequency=1,
                log_frequency=1,
                metrics_=MockMetrics()
            )

            train_losses, best_val_loss = trainer()

            assert len(train_losses) <= 2
            assert isinstance(best_val_loss, float)


class TestSampleLDM:
    """Test suite for LDM sampler."""

    @pytest.fixture
    def sampling_setup(self):
        """Setup sampling components."""
        compressor = AutoencoderLDM(
            in_channels=3, down_channels=[8, 16], up_channels=[16, 8],
            out_channels=3, dropout_rate=0.1, latent_channels=4,
            num_heads=1, num_groups=8, num_layers_per_block=2,
            total_down_sampling_factor=2, num_embeddings=32
        )

        noise_predictor = MockNoisePredictor(in_channels=4)
        text_encoder = MockTextEncoder(output_dim=32)

        scheduler = VarianceSchedulerSDE(num_steps=50, trainable_beta=False)
        reverse_sde = ReverseSDE(scheduler, "ode")

        return {
            'compressor': compressor,
            'noise_predictor': noise_predictor,
            'text_encoder': text_encoder,
            'reverse_sde': reverse_sde
        }

    def test_sample_ldm_initialization(self, sampling_setup):
        """Test SampleLDM initialization."""
        setup = sampling_setup

        sampler = SampleLDM(
            diffusion_model="sde",
            reverse_diffusion=setup['reverse_sde'],
            noise_predictor=setup['noise_predictor'],
            compressor_model=setup['compressor'],
            image_shape=(32, 32),
            conditional_model=setup['text_encoder'],
            batch_size=2,
            device='cpu'
        )

        assert sampler.diffusion_model == "sde"
        assert sampler.batch_size == 2
        assert sampler.image_shape == (32, 32)

    def test_sample_ldm_tokenization(self, sampling_setup):
        """Test text tokenization in SampleLDM."""
        setup = sampling_setup

        sampler = SampleLDM(
            diffusion_model="sde",
            reverse_diffusion=setup['reverse_sde'],
            noise_predictor=setup['noise_predictor'],
            compressor_model=setup['compressor'],
            image_shape=(32, 32),
            conditional_model=setup['text_encoder'],
            batch_size=2,
            device='cpu'
        )

        # Test single prompt
        input_ids, attention_mask = sampler.tokenize("test prompt")
        assert input_ids.shape[0] == 1

        # Test multiple prompts
        input_ids, attention_mask = sampler.tokenize(["prompt1", "prompt2"])
        assert input_ids.shape[0] == 2

    def test_sample_ldm_generation(self, sampling_setup):
        """Test image generation with SampleLDM."""
        setup = sampling_setup

        with tempfile.TemporaryDirectory() as temp_dir:
            sampler = SampleLDM(
                diffusion_model="sde",
                reverse_diffusion=setup['reverse_sde'],
                noise_predictor=setup['noise_predictor'],
                compressor_model=setup['compressor'],
                image_shape=(32, 32),
                conditional_model=setup['text_encoder'],
                batch_size=2,
                device='cpu'
            )

            # Test unconditional generation
            images = sampler(
                conditions=None,
                normalize_output=True,
                save_images=False
            )

            assert images.shape == (2, 3, 32, 32)
            assert torch.all(images >= 0) and torch.all(images <= 1)

            # Test conditional generation
            images_cond = sampler(
                conditions=["test1", "test2"],
                normalize_output=True,
                save_images=True,
                save_path=temp_dir
            )

            assert images_cond.shape == (2, 3, 32, 32)

            # Check that images were saved
            saved_files = os.listdir(temp_dir)
            assert len(saved_files) == 2


class TestIntegration:
    """Integration tests for full LDM pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete LDM pipeline from training to sampling."""
        # Create minimal dataset
        data = torch.randn(8, 3, 32, 32)
        labels = ['class_' + str(i % 2) for i in range(8)]
        dataset = TensorDataset(data, labels)
        train_loader = DataLoader(dataset, batch_size=4)

        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Train AutoEncoder
            compressor = AutoencoderLDM(
                in_channels=3, down_channels=[8, 16], up_channels=[16, 8],
                out_channels=3, dropout_rate=0.1, latent_channels=4,
                num_heads=1, num_groups=8, num_layers_per_block=2,
                total_down_sampling_factor=2, num_embeddings=32
            )

            ae_optimizer = torch.optim.Adam(compressor.parameters(), lr=1e-3)
            ae_trainer = TrainAE(
                model=compressor,
                optimizer=ae_optimizer,
                data_loader=train_loader,
                max_epochs=1,
                device='cpu',
                store_path=temp_dir
            )

            ae_losses, ae_best_loss = ae_trainer()
            assert len(ae_losses) == 1

            # 2. Train LDM
            noise_predictor = MockNoisePredictor(in_channels=4)
            text_encoder = MockTextEncoder(output_dim=32)

            scheduler = VarianceSchedulerSDE(num_steps=20, trainable_beta=False)
            forward_sde = ForwardSDE(scheduler, "ode")
            reverse_sde = ReverseSDE(scheduler, "ode")

            ldm_optimizer = torch.optim.Adam([
                *noise_predictor.parameters(),
                *text_encoder.parameters()
            ], lr=1e-3)

            ldm_trainer = TrainLDM(
                diffusion_model="sde",
                forward_diffusion=forward_sde,
                reverse_diffusion=reverse_sde,
                noise_predictor=noise_predictor,
                compressor_model=compressor,
                optimizer=ldm_optimizer,
                objective=nn.MSELoss(),
                data_loader=train_loader,
                conditional_model=text_encoder,
                max_epochs=1,
                device='cpu',
                store_path=temp_dir
            )

            ldm_losses, ldm_best_loss = ldm_trainer()
            assert len(ldm_losses) == 1

            # 3. Sample from trained model
            sampler = SampleLDM(
                diffusion_model="sde",
                reverse_diffusion=reverse_sde,
                noise_predictor=noise_predictor,
                compressor_model=compressor,
                image_shape=(32, 32),
                conditional_model=text_encoder,
                batch_size=2,
                device='cpu'
            )

            generated_images = sampler(
                conditions=["class_0", "class_1"],
                save_images=False
            )

            assert generated_images.shape == (2, 3, 32, 32)
            assert not torch.isnan(generated_images).any()


if __name__ == "__main__":
    import subprocess
    import sys

    try:
        import pytest
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest

    pytest.main([__file__, "-v"])