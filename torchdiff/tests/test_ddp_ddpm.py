"""
DDP implementation tests for TrainDDPM.

Tests DDP setup, model wrapping, checkpoint save/load with DDP state dicts,
and a simulated multi-process training run using gloo backend on CPU.
No multi-GPU required.
"""

import os
import sys
import time
import shutil
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from torchdiff.ddpm import SchedulerDDPM, ForwardDDPM, ReverseDDPM, TrainDDPM


class TinyDiffNet(nn.Module):
    """Minimal model for testing, matching DiffusionNetwork's forward API."""
    def __init__(self, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, channels, 3, padding=1),
        )
        # time embedding – used in forward to ensure all params participate in loss
        self.time_embed = nn.Linear(1, 16)

    def forward(self, x, t, y=None, clip_embeddings=None):
        # project t and add bias to feature maps so time_embed params receive grad
        t_emb = self.time_embed(t.float().unsqueeze(-1))  # [B, 16]
        h = self.net[0](x)  # Conv -> [B, 16, H, W]
        h = h + t_emb.unsqueeze(-1).unsqueeze(-1)  # broadcast add
        h = self.net[1](h)  # ReLU
        h = self.net[2](h)  # Conv -> [B, C, H, W]
        return h


def make_components(device='cpu', time_steps=10):
    """Create minimal DDPM components for testing."""
    scheduler = SchedulerDDPM(schedule_type="linear", time_steps=time_steps)
    fwd = ForwardDDPM(scheduler, pred_type="noise")
    rwd = ReverseDDPM(scheduler, pred_type="noise", var_type="fixed_small")
    model = TinyDiffNet(channels=1)
    return scheduler, fwd, rwd, model


def make_dataset(n_samples=32, channels=1, img_size=8):
    """Create a small dummy dataset."""
    x = torch.randn(n_samples, channels, img_size, img_size)
    y = torch.zeros(n_samples, dtype=torch.long)  # dummy labels
    return TensorDataset(x, y)


class TestDDPSetup:
    """Test DDP setup & teardown logic without starting a process group."""

    def test_single_gpu_setup(self):
        """Test _setup_single_gpu sets correct defaults."""
        scheduler, fwd, rwd, model = make_components()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = TrainDDPM(
            diff_net=model, fwd_ddpm=fwd, rwd_ddpm=rwd,
            train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
            device='cpu', use_ddp=False, max_epochs=1
        )
        assert trainer.ddp_rank == 0
        assert trainer.ddp_local_rank == 0
        assert trainer.ddp_world_size == 1
        assert trainer.master_process is True

    def test_ddp_setup_missing_env_vars(self):
        """Test DDP setup raises when env vars are missing."""
        scheduler, fwd, rwd, model = make_components()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Clear env vars
        for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']:
            os.environ.pop(var, None)

        with pytest.raises(ValueError, match="RANK"):
            TrainDDPM(
                diff_net=model, fwd_ddpm=fwd, rwd_ddpm=rwd,
                train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
                device='cpu', use_ddp=True, max_epochs=1
            )


class TestDDPModelWrapping:
    """Test DDP model wrapping logic."""

    def test_wrap_models_not_ddp(self):
        """Models should not be wrapped when use_ddp=False."""
        scheduler, fwd, rwd, model = make_components()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = TrainDDPM(
            diff_net=model, fwd_ddpm=fwd, rwd_ddpm=rwd,
            train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
            device='cpu', use_ddp=False, max_epochs=1
        )
        trainer._wrap_models_for_ddp()
        # Should not be wrapped
        assert not hasattr(trainer.diff_net, 'module')


class TestDDPCheckpoint:
    """Test checkpoint saving/loading with DDP state dict key handling."""

    def test_checkpoint_save_load_non_ddp(self):
        """Test checkpoint round-trip without DDP."""
        scheduler, fwd, rwd, model = make_components()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TrainDDPM(
                diff_net=model, fwd_ddpm=fwd, rwd_ddpm=rwd,
                train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
                device='cpu', use_ddp=False, max_epochs=1,
                store_path=tmpdir
            )
            # Save checkpoint
            trainer._save_checkpoint(epoch=1, loss=0.5, pref="test_")

            # Create fresh model and load
            _, fwd2, rwd2, model2 = make_components()
            optim2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            trainer2 = TrainDDPM(
                diff_net=model2, fwd_ddpm=fwd2, rwd_ddpm=rwd2,
                train_loader=loader, optim=optim2, loss_fn=nn.MSELoss(),
                device='cpu', use_ddp=False, max_epochs=1,
                store_path=tmpdir
            )
            epoch, loss = trainer2.load_checkpoint(os.path.join(tmpdir, "test_model_epoch_1.pth"))
            assert epoch == 1
            assert loss == 0.5

            # Verify model weights match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)

    def test_checkpoint_module_prefix_stripping(self):
        """Test that 'module.' prefix is properly stripped when loading DDP checkpoint into non-DDP model."""
        scheduler, fwd, rwd, model = make_components()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate a DDP checkpoint by adding 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in model.state_dict().items()}
            checkpoint = {
                'epoch': 5,
                'model_state_dict_diff_net': state_dict,
                'model_state_dict_cond': None,
                'optim_state_dict': optim.state_dict(),
                'loss': 0.3,
                'losses': {'train_losses': [], 'val_losses': []},
                'scheduler_model': fwd.vs.state_dict(),
                'max_epochs': 10,
            }
            ckpt_path = os.path.join(tmpdir, "ddp_ckpt.pth")
            torch.save(checkpoint, ckpt_path)

            # Load into non-DDP trainer
            _, fwd2, rwd2, model2 = make_components()
            optim2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            trainer = TrainDDPM(
                diff_net=model2, fwd_ddpm=fwd2, rwd_ddpm=rwd2,
                train_loader=loader, optim=optim2, loss_fn=nn.MSELoss(),
                device='cpu', use_ddp=False, max_epochs=1,
                store_path=tmpdir
            )
            epoch, loss = trainer.load_checkpoint(ckpt_path)
            assert epoch == 5
            assert loss == 0.3


class TestTrainingPipeline:
    """Test training pipeline improvements (non-DDP)."""

    def test_device_type_detection(self):
        """Test that _device_type is correctly set."""
        scheduler, fwd, rwd, model = make_components()
        dataset = make_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = TrainDDPM(
            diff_net=model, fwd_ddpm=fwd, rwd_ddpm=rwd,
            train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
            device='cpu', use_ddp=False, max_epochs=1
        )
        assert trainer._device_type == 'cpu'

    def test_training_runs_on_cpu(self):
        """Test that training loop completes on CPU with AMP disabled."""
        scheduler, fwd, rwd, model = make_components(time_steps=5)
        dataset = make_dataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = TrainDDPM(
            diff_net=model, fwd_ddpm=fwd, rwd_ddpm=rwd,
            train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
            device='cpu', use_ddp=False, max_epochs=2, warmup_steps=2,
            log_freq=1
        )
        losses = trainer()
        assert 'train_losses' in losses
        assert len(losses['train_losses']) == 2
        assert all(isinstance(l, float) for l in losses['train_losses'])

    def test_gradient_accumulation(self):
        """Test gradient accumulation with grad_acc > 1."""
        scheduler, fwd, rwd, model = make_components(time_steps=5)
        dataset = make_dataset(n_samples=16)
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = TrainDDPM(
            diff_net=model, fwd_ddpm=fwd, rwd_ddpm=rwd,
            train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
            device='cpu', use_ddp=False, max_epochs=1,
            grad_acc=2, warmup_steps=1
        )
        losses = trainer()
        assert len(losses['train_losses']) == 1

    def test_warmup_scheduler_starts_at_step_zero(self):
        """Test that warmup scheduler steps from step 0 (not step 1)."""
        scheduler, fwd, rwd, model = make_components(time_steps=5)
        dataset = make_dataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = TrainDDPM(
            diff_net=model, fwd_ddpm=fwd, rwd_ddpm=rwd,
            train_loader=loader, optim=optim, loss_fn=nn.MSELoss(),
            device='cpu', use_ddp=False, max_epochs=1,
            warmup_steps=100
        )
        # Before training, global_step should be 0
        assert trainer.global_step == 0

        # After training 1 epoch with 2 batches (8 samples / 4 batch_size)
        trainer()
        assert trainer.global_step == 2


class TestDDPSpawn:
    """Test actual DDP training using gloo backend on CPU with multiprocessing.spawn."""

    @staticmethod
    def _ddp_worker(rank, world_size, tmpdir, results_dict):
        """Worker function for DDP training."""
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501'

        try:
            import torch.distributed as dist
            dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

            scheduler = SchedulerDDPM(schedule_type="linear", time_steps=5)
            fwd = ForwardDDPM(scheduler, pred_type="noise")
            rwd = ReverseDDPM(scheduler, pred_type="noise", var_type="fixed_small")
            model = TinyDiffNet(channels=1)

            dataset = make_dataset(n_samples=16, channels=1, img_size=8)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            loader = DataLoader(dataset, batch_size=4, sampler=sampler)

            optim = torch.optim.Adam(model.parameters(), lr=1e-3)

            # We need to bypass the _setup_ddp since we already initialized
            # Instead, manually set attributes
            trainer = TrainDDPM.__new__(TrainDDPM)
            nn.Module.__init__(trainer)

            trainer.use_ddp = True
            trainer.grad_acc = 1
            trainer.device = torch.device('cpu')
            trainer.ddp_rank = rank
            trainer.ddp_local_rank = rank
            trainer.ddp_world_size = world_size
            trainer.master_process = (rank == 0)

            trainer.diff_net = model.to(trainer.device)
            trainer.fwd_ddpm = fwd.to(trainer.device)
            trainer.rwd_ddpm = rwd.to(trainer.device)
            trainer.cond_net = None
            trainer.metrics_ = None
            trainer.optim = optim
            from torchdiff.utils import LossAdapter
            trainer.loss_fn = LossAdapter(nn.MSELoss())
            trainer.store_path = tmpdir
            trainer.train_loader = loader
            trainer.val_loader = None
            trainer.max_epochs = 2
            trainer.max_token_length = 77
            trainer.patience = 20
            trainer.val_freq = 10
            trainer.norm_range = (-1.0, 1.0)
            trainer.norm_output = True
            trainer.log_freq = 1
            trainer.use_comp = False
            trainer.global_step = 0
            trainer.warmup_steps = 100
            trainer.best_loss = float('inf')
            trainer.losses = {'train_losses': [], 'val_losses': []}
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            trainer.scheduler = ReduceLROnPlateau(optim, patience=20, factor=0.5)
            trainer.warmup_lr_scheduler = TrainDDPM.warmup_scheduler(optim, 100)
            trainer._device_type = 'cpu'

            from transformers import BertTokenizer
            trainer.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            losses = trainer()

            results_dict[rank] = {
                'train_losses': losses['train_losses'],
                'success': True,
                'num_epochs': len(losses['train_losses']),
            }

        except Exception as e:
            results_dict[rank] = {
                'success': False,
                'error': str(e),
            }
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            # Clean up env vars
            for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
                os.environ.pop(var, None)

    def test_ddp_two_processes_gloo(self):
        """Test DDP training with 2 processes using gloo backend on CPU."""
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = mp.Manager()
            results = manager.dict()

            mp.spawn(
                self._ddp_worker,
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
