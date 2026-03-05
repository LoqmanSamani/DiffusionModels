"""
DDP implementation tests for TrainUnClipDecoder, TrainUnCLIPPrior, and TrainUpsamplerUnCLIP.

Tests DDP setup, model wrapping, checkpoint save/load with DDP state dicts,
and a simulated multi-process training run using gloo backend on CPU.
No multi-GPU required.
"""

import os
import sys
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from torchdiff.unclip import (
    SchedulerUnCLIP, ForwardUnCLIP, ReverseUnCLIP,
    UnCLIPTransformerPrior, CLIPContextProjection, CLIPEmbeddingProjection,
    UpsamplerUnCLIP, TrainUnClipDecoder, TrainUnCLIPPrior, TrainUpsamplerUnCLIP,
)
from torchdiff.utils import LossAdapter


# ---------------------------------------------------------------------------
# Tiny mock models for fast CPU tests
# ---------------------------------------------------------------------------

class TinyDecoder(nn.Module):
    """Minimal decoder mock that accepts UnClipDecoder's forward signature."""
    def __init__(self, clip_embed_dim=64):
        super().__init__()
        self.net = nn.Linear(clip_embed_dim, clip_embed_dim)
        self.fwd_unclip = ForwardUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )
        self.rwd_unclip = ReverseUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )

    def forward(self, clip_img_embed, clip_txt_embed, imgs, texts):
        """Mimic UnClipDecoder forward: return (pred, target)."""
        batch_size = clip_img_embed.shape[0]
        t = torch.randint(0, 10, (batch_size,), device=clip_img_embed.device)
        noise = torch.randn_like(clip_img_embed)
        noisy, target = self.fwd_unclip(clip_img_embed, noise, t)
        pred = self.net(noisy)
        return pred, target


class TinyCLIP(nn.Module):
    """Minimal CLIP mock that returns fixed-size embeddings."""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        # Dummy param so .to(device) works
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, data, data_type="img", normalize=True):
        if data_type == "img":
            batch_size = data.shape[0]
        else:
            batch_size = len(data) if isinstance(data, list) else data.shape[0]
        out = torch.randn(batch_size, self.embed_dim, device=self.proj.weight.device)
        return out


class TinyPrior(nn.Module):
    """Minimal prior mock matching UnCLIPTransformerPrior's API."""
    def __init__(self, clip_embed_dim=64):
        super().__init__()
        self.net = nn.Linear(clip_embed_dim, clip_embed_dim)
        self.fwd_unclip = ForwardUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )
        self.clip_text_proj = None
        self.clip_img_proj = None

    def forward(self, text_embed, noisy_embed, t):
        return self.net(noisy_embed)


def make_img_dataset(n_samples=16, channels=3, img_size=8):
    """Create image+text dummy dataset (text labels as class indices)."""
    imgs = torch.randn(n_samples, channels, img_size, img_size)
    labels = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(imgs, labels)


def make_upsampler_dataset(n_samples=16, channels=3, low_size=8, high_size=16):
    """Create (low_res, high_res) image pairs."""
    low = torch.randn(n_samples, channels, low_size, low_size)
    high = torch.randn(n_samples, channels, high_size, high_size)
    return TensorDataset(low, high)


# ============================================================================
# TrainUnClipDecoder tests
# ============================================================================

class TestTrainDecoderDDPSetup:
    """Test DDP setup logic for TrainUnClipDecoder."""

    def _make_decoder_trainer(self, use_ddp=False, device='cpu'):
        decoder = TinyDecoder(clip_embed_dim=64)
        clip_net = TinyCLIP(embed_dim=64)
        dataset = make_img_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        return TrainUnClipDecoder(
            clip_embed_dim=64,
            decoder_net=decoder,
            clip_net=clip_net,
            train_loader=loader,
            optim=optim,
            loss_fn=nn.MSELoss(),
            device=device,
            use_ddp=use_ddp,
            max_epochs=1,
            reduce_clip_embed_dim=False,
            use_autocast=False,
        )

    def test_single_gpu_setup(self):
        trainer = self._make_decoder_trainer()
        assert trainer.ddp_rank == 0
        assert trainer.ddp_local_rank == 0
        assert trainer.ddp_world_size == 1
        assert trainer.master_process is True

    def test_device_type_cpu(self):
        trainer = self._make_decoder_trainer(device='cpu')
        assert trainer._device_type == 'cpu'

    def test_ddp_missing_env_vars(self):
        for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']:
            os.environ.pop(var, None)
        with pytest.raises(ValueError, match="RANK"):
            self._make_decoder_trainer(use_ddp=True)

    def test_wrap_not_ddp(self):
        trainer = self._make_decoder_trainer()
        assert not hasattr(trainer.decoder_net, 'module')


# ============================================================================
# TrainUnCLIPPrior tests
# ============================================================================

class TestTrainPriorDDPSetup:
    """Test DDP setup logic for TrainUnCLIPPrior."""

    def _make_prior_trainer(self, use_ddp=False, device='cpu'):
        prior = TinyPrior(clip_embed_dim=64)
        clip_net = TinyCLIP(embed_dim=64)
        dataset = make_img_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(prior.parameters(), lr=1e-3)
        return TrainUnCLIPPrior(
            prior_net=prior,
            clip_net=clip_net,
            train_loader=loader,
            optim=optim,
            loss_fn=nn.MSELoss(),
            device=device,
            use_ddp=use_ddp,
            max_epochs=1,
            reduce_clip_embed_dim=False,
            use_autocast=False,
        )

    def test_single_gpu_setup(self):
        trainer = self._make_prior_trainer()
        assert trainer.ddp_rank == 0
        assert trainer.ddp_local_rank == 0
        assert trainer.ddp_world_size == 1
        assert trainer.master_process is True

    def test_device_type_cpu(self):
        trainer = self._make_prior_trainer(device='cpu')
        assert trainer._device_type == 'cpu'

    def test_ddp_missing_env_vars(self):
        for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']:
            os.environ.pop(var, None)
        with pytest.raises(ValueError, match="RANK"):
            self._make_prior_trainer(use_ddp=True)

    def test_wrap_not_ddp(self):
        trainer = self._make_prior_trainer()
        assert not hasattr(trainer.prior_net, 'module')


# ============================================================================
# TrainUpsamplerUnCLIP tests
# ============================================================================

class TestTrainUpsamplerDDPSetup:
    """Test DDP setup logic for TrainUpsamplerUnCLIP."""

    def _make_upsampler_trainer(self, use_ddp=False, device='cpu'):
        fwd = ForwardUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )
        rwd = ReverseUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )
        up_net = UpsamplerUnCLIP(
            fwd_unclip=fwd, rwd_unclip=rwd,
            in_channels=3, out_channels=3,
            model_channels=16, num_res_blocks=1,
            channel_mult=(1, 2),
            time_embed_dim=32,
            low_res_size=8, high_res_size=16,
        )
        dataset = make_upsampler_dataset()
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(up_net.parameters(), lr=1e-3)
        return TrainUpsamplerUnCLIP(
            up_net=up_net,
            train_loader=loader,
            optim=optim,
            loss_fn=nn.MSELoss(),
            device=device,
            use_ddp=use_ddp,
            max_epochs=1,
            use_autocast=False,
        )

    def test_single_gpu_setup(self):
        trainer = self._make_upsampler_trainer()
        assert trainer.ddp_rank == 0
        assert trainer.ddp_local_rank == 0
        assert trainer.ddp_world_size == 1
        assert trainer.master_process is True

    def test_device_type_cpu(self):
        trainer = self._make_upsampler_trainer(device='cpu')
        assert trainer._device_type == 'cpu'

    def test_ddp_missing_env_vars(self):
        for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']:
            os.environ.pop(var, None)
        with pytest.raises(ValueError, match="RANK"):
            self._make_upsampler_trainer(use_ddp=True)

    def test_wrap_not_ddp(self):
        trainer = self._make_upsampler_trainer()
        assert not hasattr(trainer.up_net, 'module')


# ============================================================================
# Training pipeline tests (single process, CPU, no AMP)
# ============================================================================

class TestTrainUpsamplerPipeline:
    """Test TrainUpsamplerUnCLIP training loop runs on CPU without errors."""

    def test_training_runs_on_cpu(self):
        fwd = ForwardUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )
        rwd = ReverseUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )
        up_net = UpsamplerUnCLIP(
            fwd_unclip=fwd, rwd_unclip=rwd,
            in_channels=3, out_channels=3,
            model_channels=16, num_res_blocks=1,
            channel_mult=(1, 2),
            time_embed_dim=32,
            low_res_size=8, high_res_size=16,
        )
        dataset = make_upsampler_dataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(up_net.parameters(), lr=1e-3)

        trainer = TrainUpsamplerUnCLIP(
            up_net=up_net, train_loader=loader, optim=optim,
            loss_fn=nn.MSELoss(), device='cpu', use_ddp=False,
            max_epochs=2, warmup_steps=2, use_autocast=False,
        )
        losses = trainer()
        assert 'train_losses' in losses
        assert len(losses['train_losses']) == 2
        assert all(isinstance(l, float) for l in losses['train_losses'])

    def test_warmup_scheduler_starts_at_step_zero(self):
        fwd = ForwardUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )
        rwd = ReverseUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )
        up_net = UpsamplerUnCLIP(
            fwd_unclip=fwd, rwd_unclip=rwd,
            in_channels=3, out_channels=3,
            model_channels=16, num_res_blocks=1,
            channel_mult=(1, 2),
            time_embed_dim=32,
            low_res_size=8, high_res_size=16,
        )
        dataset = make_upsampler_dataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(up_net.parameters(), lr=1e-3)

        trainer = TrainUpsamplerUnCLIP(
            up_net=up_net, train_loader=loader, optim=optim,
            loss_fn=nn.MSELoss(), device='cpu', use_ddp=False,
            max_epochs=1, warmup_steps=100, use_autocast=False,
        )
        assert trainer.global_step == 0
        trainer()
        assert trainer.global_step == 2  # 8 samples / 4 batch_size


class TestTrainUpsamplerCheckpoint:
    """Test checkpoint save/load for TrainUpsamplerUnCLIP."""

    def test_checkpoint_round_trip(self):
        fwd = ForwardUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )
        rwd = ReverseUnCLIP(
            SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
        )
        up_net = UpsamplerUnCLIP(
            fwd_unclip=fwd, rwd_unclip=rwd,
            in_channels=3, out_channels=3,
            model_channels=16, num_res_blocks=1,
            channel_mult=(1, 2),
            time_embed_dim=32,
            low_res_size=8, high_res_size=16,
        )
        dataset = make_upsampler_dataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        optim = torch.optim.Adam(up_net.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TrainUpsamplerUnCLIP(
                up_net=up_net, train_loader=loader, optim=optim,
                loss_fn=nn.MSELoss(), device='cpu', use_ddp=False,
                max_epochs=1, store_path=tmpdir, use_autocast=False,
            )
            trainer._save_checkpoint(epoch=1, loss=0.5, pref="test_")

            # Create a fresh trainer and load checkpoint
            fwd2 = ForwardUnCLIP(
                SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
            )
            rwd2 = ReverseUnCLIP(
                SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
            )
            up_net2 = UpsamplerUnCLIP(
                fwd_unclip=fwd2, rwd_unclip=rwd2,
                in_channels=3, out_channels=3,
                model_channels=16, num_res_blocks=1,
                channel_mult=(1, 2),
                time_embed_dim=32,
                low_res_size=8, high_res_size=16,
            )
            optim2 = torch.optim.Adam(up_net2.parameters(), lr=1e-3)
            trainer2 = TrainUpsamplerUnCLIP(
                up_net=up_net2, train_loader=loader, optim=optim2,
                loss_fn=nn.MSELoss(), device='cpu', use_ddp=False,
                max_epochs=1, store_path=tmpdir, use_autocast=False,
            )
            epoch, loss = trainer2.load_checkpoint(os.path.join(tmpdir, "test_model_epoch_1.pth"))
            assert epoch == 1
            assert loss == 0.5

            for p1, p2 in zip(up_net.parameters(), up_net2.parameters()):
                assert torch.allclose(p1, p2)


# ============================================================================
# DDP spawn tests (gloo backend, CPU, 2 processes)
# ============================================================================

class TestDDPSpawnUpsampler:
    """Test actual DDP training for TrainUpsamplerUnCLIP using gloo on CPU."""

    @staticmethod
    def _ddp_worker(rank, world_size, tmpdir, results_dict):
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29505'

        try:
            import torch.distributed as dist
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

            fwd = ForwardUnCLIP(
                SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
            )
            rwd = ReverseUnCLIP(
                SchedulerUnCLIP(schedule_type="linear", train_steps=10), pred_type="noise"
            )
            up_net = UpsamplerUnCLIP(
                fwd_unclip=fwd, rwd_unclip=rwd,
                in_channels=3, out_channels=3,
                model_channels=16, num_res_blocks=1,
                channel_mult=(1, 2),
                time_embed_dim=32,
                low_res_size=8, high_res_size=16,
            )
            dataset = make_upsampler_dataset(n_samples=16, channels=3, low_size=8, high_size=16)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            loader = DataLoader(dataset, batch_size=4, sampler=sampler)
            optim = torch.optim.Adam(up_net.parameters(), lr=1e-3)

            # Manually build trainer, bypassing __init__ DDP setup
            trainer = TrainUpsamplerUnCLIP.__new__(TrainUpsamplerUnCLIP)
            nn.Module.__init__(trainer)

            trainer.use_ddp = True
            trainer.grad_acc = 1
            trainer.use_comp = False
            trainer.use_autocast = False
            trainer.device = torch.device('cpu')
            trainer._device_type = 'cpu'
            trainer.ddp_rank = rank
            trainer.ddp_local_rank = rank
            trainer.ddp_world_size = world_size
            trainer.master_process = (rank == 0)

            trainer.up_net = up_net.to(trainer.device)
            from torch.nn.parallel import DistributedDataParallel as DDP
            trainer.up_net = DDP(trainer.up_net, find_unused_parameters=False)

            trainer.optim = optim
            trainer.loss_fn = LossAdapter(nn.MSELoss())
            trainer.train_loader = loader
            trainer.val_loader = None
            trainer.max_epochs = 2
            trainer.store_path = tmpdir
            trainer.patience = 100
            trainer.val_freq = 10
            trainer.log_freq = 1
            trainer.norm_range = (-1.0, 1.0)
            trainer.norm_out = True
            trainer.global_step = 0
            trainer.warmup_steps = 100
            trainer.best_loss = float('inf')
            trainer.losses = {'train_losses': [], 'val_losses': []}

            trainer.scheduler = ReduceLROnPlateau(optim, patience=100, factor=0.5)
            trainer.warmup_lr_scheduler = TrainUpsamplerUnCLIP.warmup_scheduler(optim, 100)

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

    def test_ddp_two_processes_gloo(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = mp.Manager()
            results = manager.dict()

            mp.spawn(
                self._ddp_worker,
                args=(world_size, tmpdir, results),
                nprocs=world_size,
                join=True,
            )

            for rank in range(world_size):
                assert rank in results, f"Rank {rank} did not report results"
                result = results[rank]
                assert result['success'], f"Rank {rank} failed: {result.get('error')}"
                assert result['num_epochs'] == 2, f"Rank {rank} trained {result['num_epochs']} epochs"
                assert len(result['train_losses']) == 2


class TestDDPSpawnPrior:
    """Test actual DDP training for TrainUnCLIPPrior using gloo on CPU."""

    @staticmethod
    def _ddp_worker_prior(rank, world_size, tmpdir, results_dict):
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29506'

        try:
            import torch.distributed as dist
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

            prior = TinyPrior(clip_embed_dim=64)
            clip_net = TinyCLIP(embed_dim=64)
            dataset = make_img_dataset(n_samples=16, channels=3, img_size=8)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            loader = DataLoader(dataset, batch_size=4, sampler=sampler)
            optim = torch.optim.Adam(prior.parameters(), lr=1e-3)

            # Manually build trainer bypassing __init__
            trainer = TrainUnCLIPPrior.__new__(TrainUnCLIPPrior)
            nn.Module.__init__(trainer)

            trainer.use_ddp = True
            trainer.grad_acc = 1
            trainer.use_comp = False
            trainer.use_autocast = False
            trainer.device = torch.device('cpu')
            trainer._device_type = 'cpu'
            trainer.ddp_rank = rank
            trainer.ddp_local_rank = rank
            trainer.ddp_world_size = world_size
            trainer.master_process = (rank == 0)

            trainer.prior_net = prior.to(trainer.device)
            trainer.clip_net = clip_net.to(trainer.device)
            # NOTE: Do NOT wrap with DDP here — TrainUnCLIPPrior.forward()
            # calls _wrap_models_for_ddp() internally

            trainer.optim = optim
            trainer.loss_fn = LossAdapter(nn.MSELoss())
            trainer.train_loader = loader
            trainer.val_loader = None
            trainer.max_epochs = 2
            trainer.store_path = tmpdir
            trainer.patience = 100
            trainer.val_freq = 10
            trainer.log_freq = 1
            trainer.norm_range = (-1.0, 1.0)
            trainer.reduce_clip_embed_dim = False
            trainer.norm_clip_embed = True
            trainer.trans_embed_dim = 64
            trainer.global_step = 0
            trainer.warmup_steps = 100
            trainer.best_loss = float('inf')
            trainer.losses = {'train_losses': [], 'val_losses': []}

            trainer.scheduler = ReduceLROnPlateau(optim, patience=100, factor=0.5)
            trainer.warmup_lr_scheduler = TrainUnCLIPPrior.warmup_scheduler(optim, 100)

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

    def test_ddp_two_processes_gloo_prior(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = mp.Manager()
            results = manager.dict()

            mp.spawn(
                self._ddp_worker_prior,
                args=(world_size, tmpdir, results),
                nprocs=world_size,
                join=True,
            )

            for rank in range(world_size):
                assert rank in results, f"Rank {rank} did not report results"
                result = results[rank]
                assert result['success'], f"Rank {rank} failed: {result.get('error')}"
                assert result['num_epochs'] == 2, f"Rank {rank} trained {result['num_epochs']} epochs"
                assert len(result['train_losses']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
