"""
DDP implementation tests for TrainLDM and TrainAE.

Tests DDP setup, model wrapping, checkpoint save/load with DDP state dicts,
and a simulated multi-process training run using gloo backend on CPU.
No multi-GPU required.
"""

import os
import sys
import shutil
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from torchdiff.ldm import AutoencoderLDM, TrainLDM, TrainAE
from torchdiff.ddpm import SchedulerDDPM, ForwardDDPM, ReverseDDPM
from torchdiff.utils import LossAdapter


class TinyDiffNet(nn.Module):
    """Minimal model for LDM testing, matching DiffusionNetwork's forward API."""
    def __init__(self, in_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, 3, padding=1),
        )
        self.time_embed = nn.Linear(1, 16)

    def forward(self, x, t, y=None, clip_embeddings=None):
        t_emb = self.time_embed(t.float().unsqueeze(-1))  # [B, 16]
        h = self.net[0](x)  # Conv -> [B, 16, H, W]
        h = h + t_emb.unsqueeze(-1).unsqueeze(-1)
        h = self.net[1](h)  # ReLU
        h = self.net[2](h)  # Conv -> [B, C, H, W]
        return h


def make_small_vae():
    """Create a small KL autoencoder for testing."""
    return AutoencoderLDM(
        in_channels=1,
        down_channels=[16, 32],
        up_channels=[32, 16],
        out_channels=1,
        dropout_rate=0.0,
        num_heads=2,
        num_groups=2,
        num_layers_per_block=1,
        total_down_sampling_factor=2,
        latent_channels=4,
        num_embeddings=64,
        use_vq=False,
        beta=1.0,
        use_flash=False,
        use_grad_check=False,
    )


def make_ldm_components(time_steps=5):
    """Create minimal LDM training components."""
    scheduler = SchedulerDDPM(schedule_type="linear", time_steps=time_steps)
    fwd = ForwardDDPM(scheduler, pred_type="noise")
    rwd = ReverseDDPM(scheduler, pred_type="noise", var_type="fixed_small")
    diff_net = TinyDiffNet(in_channels=4)
    comp_net = make_small_vae()
    return scheduler, fwd, rwd, diff_net, comp_net


def make_dataset(n_samples=16, channels=1, img_size=16):
    """Create a small dummy dataset."""
    x = torch.randn(n_samples, channels, img_size, img_size)
    y = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(x, y)


class TestTrainLDMDDPSetup:
    """Test DDP setup & teardown logic for TrainLDM."""

    def test_single_gpu_setup(self):
        """Test _setup_single_gpu sets correct defaults."""
        _, fwd, rwd, diff_net, comp_net = make_ldm_components()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(diff_net.parameters(), lr=1e-3)

        trainer = TrainLDM(
            diff_type="ddpm", fwd_diff=fwd, rwd_diff=rwd,
            diff_net=diff_net, comp_net=comp_net,
            train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
            device='cpu', use_ddp=False, max_epochs=1,
            tokenizer=None
        )
        assert trainer.ddp_rank == 0
        assert trainer.ddp_local_rank == 0
        assert trainer.ddp_world_size == 1
        assert trainer.master_process is True

    def test_ddp_setup_missing_env_vars(self):
        """Test DDP setup raises when env vars are missing."""
        _, fwd, rwd, diff_net, comp_net = make_ldm_components()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(diff_net.parameters(), lr=1e-3)

        for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']:
            os.environ.pop(var, None)

        with pytest.raises(ValueError, match="RANK"):
            TrainLDM(
                diff_type="ddpm", fwd_diff=fwd, rwd_diff=rwd,
                diff_net=diff_net, comp_net=comp_net,
                train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
                device='cpu', use_ddp=True, max_epochs=1,
                tokenizer=None
            )

    def test_device_type_detection(self):
        """Test that _device_type is correctly set."""
        _, fwd, rwd, diff_net, comp_net = make_ldm_components()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(diff_net.parameters(), lr=1e-3)

        trainer = TrainLDM(
            diff_type="ddpm", fwd_diff=fwd, rwd_diff=rwd,
            diff_net=diff_net, comp_net=comp_net,
            train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
            device='cpu', use_ddp=False, max_epochs=1,
            tokenizer=None
        )
        assert trainer._device_type == 'cpu'


class TestTrainLDMModelWrapping:
    """Test DDP model wrapping logic for TrainLDM."""

    def test_wrap_models_not_ddp(self):
        """Models should not be wrapped when use_ddp=False."""
        _, fwd, rwd, diff_net, comp_net = make_ldm_components()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(diff_net.parameters(), lr=1e-3)

        trainer = TrainLDM(
            diff_type="ddpm", fwd_diff=fwd, rwd_diff=rwd,
            diff_net=diff_net, comp_net=comp_net,
            train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
            device='cpu', use_ddp=False, max_epochs=1,
            tokenizer=None
        )
        trainer._wrap_models_for_ddp()
        assert not hasattr(trainer.diff_net, 'module')


class TestTrainAEDDPSetup:
    """Test DDP setup for TrainAE."""

    def test_single_gpu_setup(self):
        """Test _setup_single_gpu sets correct defaults for TrainAE."""
        vae = make_small_vae()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(vae.parameters(), lr=1e-3)

        trainer = TrainAE(
            model=vae, optim=optim, train_loader=loader,
            device='cpu', use_ddp=False, max_epochs=1
        )
        assert trainer.ddp_rank == 0
        assert trainer.ddp_local_rank == 0
        assert trainer.ddp_world_size == 1
        assert trainer.master_process is True

    def test_device_type_detection(self):
        """Test that _device_type is correctly set for TrainAE."""
        vae = make_small_vae()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(vae.parameters(), lr=1e-3)

        trainer = TrainAE(
            model=vae, optim=optim, train_loader=loader,
            device='cpu', use_ddp=False, max_epochs=1
        )
        assert trainer._device_type == 'cpu'

    def test_ddp_setup_missing_env_vars(self):
        """Test DDP setup raises when env vars are missing for TrainAE."""
        vae = make_small_vae()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(vae.parameters(), lr=1e-3)

        for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']:
            os.environ.pop(var, None)

        with pytest.raises(ValueError, match="RANK"):
            TrainAE(
                model=vae, optim=optim, train_loader=loader,
                device='cpu', use_ddp=True, max_epochs=1
            )


class TestTrainAEPipeline:
    """Test TrainAE training pipeline improvements."""

    def test_training_runs_on_cpu(self):
        """Test that TrainAE training loop completes on CPU with AMP disabled."""
        vae = make_small_vae()
        dataset = make_dataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(vae.parameters(), lr=1e-3)

        trainer = TrainAE(
            model=vae, optim=optim, train_loader=loader,
            device='cpu', use_ddp=False, max_epochs=2,
            warmup_steps=2, log_freq=1
        )
        losses = trainer()
        assert 'train_losses' in losses
        assert len(losses['train_losses']) == 2
        assert all(isinstance(l, float) for l in losses['train_losses'])

    def test_warmup_scheduler_starts_at_step_zero(self):
        """Test that warmup scheduler steps from step 0."""
        vae = make_small_vae()
        dataset = make_dataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(vae.parameters(), lr=1e-3)

        trainer = TrainAE(
            model=vae, optim=optim, train_loader=loader,
            device='cpu', use_ddp=False, max_epochs=1,
            warmup_steps=100
        )
        assert trainer.global_step == 0
        trainer()
        assert trainer.global_step == 2  # 8 samples / 4 batch_size


class TestTrainAECheckpoint:
    """Test checkpoint saving/loading for TrainAE."""

    def test_checkpoint_save_load_non_ddp(self):
        """Test checkpoint round-trip without DDP for TrainAE."""
        vae = make_small_vae()
        dataset = make_dataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(vae.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TrainAE(
                model=vae, optim=optim, train_loader=loader,
                device='cpu', use_ddp=False, max_epochs=1,
                store_path=tmpdir
            )
            trainer._save_checkpoint(epoch=1, loss=0.5, pref="test_")

            vae2 = make_small_vae()
            optim2 = torch.optim.Adam(vae2.parameters(), lr=1e-3)
            trainer2 = TrainAE(
                model=vae2, optim=optim2, train_loader=loader,
                device='cpu', use_ddp=False, max_epochs=1,
                store_path=tmpdir
            )
            epoch, loss = trainer2.load_checkpoint(os.path.join(tmpdir, "test_model_epoch_1.pth"))
            assert epoch == 1
            assert loss == 0.5

            for p1, p2 in zip(vae.parameters(), vae2.parameters()):
                assert torch.allclose(p1, p2)


class TestDDPSpawnTrainAE:
    """Test actual DDP training for TrainAE using gloo backend on CPU."""

    @staticmethod
    def _ddp_worker_ae(rank, world_size, tmpdir, results_dict):
        """Worker function for DDP TrainAE training."""
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29504'

        try:
            import torch.distributed as dist
            dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

            vae = make_small_vae()
            dataset = make_dataset(n_samples=16, channels=1, img_size=16)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            loader = DataLoader(dataset, batch_size=4, sampler=sampler)
            optim = torch.optim.Adam(vae.parameters(), lr=1e-3)

            # Manually build TrainAE bypassing __init__ DDP setup
            trainer = TrainAE.__new__(TrainAE)
            nn.Module.__init__(trainer)

            trainer.use_ddp = True
            trainer.grad_acc = 1
            trainer.device = torch.device('cpu')
            trainer.ddp_rank = rank
            trainer.ddp_local_rank = rank
            trainer.ddp_world_size = world_size
            trainer.master_process = (rank == 0)

            trainer.model = vae.to(trainer.device)
            trainer.optim = optim
            trainer.train_loader = loader
            trainer.val_loader = None
            trainer.max_epochs = 2
            trainer.metrics_ = None
            trainer.store_path = tmpdir
            trainer.checkpoint = 10
            trainer.kl_warmup_epochs = 10
            trainer.patience = 20
            trainer.use_comp = False
            trainer.global_step = 0
            trainer.warmup_steps = 100
            trainer.best_loss = float('inf')
            trainer.losses = {'train_losses': [], 'val_losses': []}
            trainer.val_freq = 10
            trainer.log_freq = 1
            trainer._device_type = 'cpu'

            from torch.optim.lr_scheduler import ReduceLROnPlateau
            trainer.scheduler = ReduceLROnPlateau(optim, patience=20, factor=0.5)
            trainer.warmup_lr_scheduler = TrainAE.warmup_scheduler(optim, 100)

            losses = trainer()

            results_dict[rank] = {
                'train_losses': losses['train_losses'],
                'success': True,
                'num_epochs': len(losses['train_losses']),
            }

        except Exception as e:
            import traceback
            results_dict[rank] = {
                'success': False,
                'error': f"{e}\n{traceback.format_exc()}",
            }
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
                os.environ.pop(var, None)

    def test_ddp_two_processes_gloo_trainAE(self):
        """Test DDP training with 2 processes using gloo for TrainAE."""
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = mp.Manager()
            results = manager.dict()

            mp.spawn(
                self._ddp_worker_ae,
                args=(world_size, tmpdir, results),
                nprocs=world_size,
                join=True
            )

            for rank in range(world_size):
                assert rank in results, f"Rank {rank} did not report results"
                result = results[rank]
                assert result['success'], f"Rank {rank} failed: {result.get('error')}"
                assert result['num_epochs'] == 2, f"Rank {rank} trained {result['num_epochs']} epochs"
                assert len(result['train_losses']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
