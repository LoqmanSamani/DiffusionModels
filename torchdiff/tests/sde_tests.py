import torch
import torch.nn as nn
import unittest
from torch.utils.data import DataLoader, Dataset
from sde import ForwardSDE, ReverseSDE, HyperParamsSDE, TrainSDE, SampleSDE
from utils import TextEncoder, NoisePredictor, Metrics
import os
import numpy as np


class MockDataset(Dataset):

    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, 64, 64)
        self.labels = [f"label_{i}" for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class TestSDE(unittest.TestCase):
    pass

    def test_hyper_params(self):
        num_steps = 1000
        beta_start = 1e-4
        beta_end = 0.02
        sigma_start = 1e-3
        sigma_end = 10.0
        methods = ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]

        for method in methods:
            print(f"  Beta method: {method}")
            hp = HyperParamsSDE(
                num_steps=num_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                trainable_beta=False,
                beta_method=method,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                start=0.0,
                end=1.0
            )

            assert hp.betas.shape == (num_steps,), f"Beta shape mismatch: {hp.betas.shape}"
            assert torch.all(hp.betas >= beta_start), f"Betas below {beta_start}"
            assert torch.all(hp.betas <= beta_end), f"Betas above {beta_end}"
            assert torch.all(hp.betas.isfinite()), "Betas contain NaN/Inf"

            assert hp.cum_betas.shape == (num_steps,), f"Cum_betas shape mismatch: {hp.cum_betas.shape}"
            assert torch.all(hp.cum_betas >= 0), "Negative cum_betas"
            assert torch.all(hp.cum_betas.isfinite()), "Cum_betas contain NaN/Inf"
            assert torch.all(hp.cum_betas[1:] >= hp.cum_betas[:-1]), "Cum_betas not increasing"

            assert hp.sigmas.shape == (num_steps,), f"Sigma shape mismatch: {hp.sigmas.shape}"
            assert torch.all(hp.sigmas >= sigma_start), f"Sigmas below {sigma_start}"
            assert torch.all(hp.sigmas <= sigma_end), f"Sigmas above {sigma_end}"
            assert torch.all(hp.sigmas.isfinite()), "Sigmas contain NaN/Inf"

            for sde_method in ["ve", "vp", "sub-vp"]:
                var = hp.get_variance(torch.tensor([num_steps - 1]), sde_method)
                assert var.shape == (1,), f"Variance shape mismatch for {sde_method}: {var.shape}"
                assert var >= 0, f"Negative variance for {sde_method}: {var}"
                if sde_method == "ve":
                    expected_var = sigma_end ** 2
                    assert abs(
                        var.item() - expected_var) < 1e-2, f"VE variance mismatch: got {var.item()}, expected {expected_var}"
                elif sde_method == "vp":
                    expected_var = 1 - np.exp(-hp.cum_betas[-1].item())
                    assert abs(
                        var.item() - expected_var) < 1e-2, f"VP variance mismatch: got {var.item()}, expected {expected_var}"
                elif sde_method == "sub-vp":
                    expected_var = 1 - np.exp(-2 * hp.cum_betas[-1].item())
                    assert abs(
                        var.item() - expected_var) < 1e-2, f"Sub-VP variance mismatch: got {var.item()}, expected {expected_var}"

        print("HyperParams tests passed!")

    def test_forward_sde(self):
        num_steps = 1000
        batch_size = 4
        channels = 3
        height = 32
        width = 32
        hp = HyperParamsSDE(
            num_steps=num_steps,
            beta_start=1e-2,
            beta_end=0.02,
            trainable_beta=False,
            beta_method="linear"
        )

        x0 = torch.rand(batch_size, channels, height, width) * 2 - 1
        x0 = x0.to(torch.float32)
        input_var = 1.0 / 3.0
        for method in ["ve", "vp", "sub-vp", "ode"]:
            print(f"  SDE method: {method}")
            sde = ForwardSDE(hp, method)
            x_t = x0.clone()
            for t in range(num_steps):
                noise = torch.randn_like(x_t)
                time_steps = torch.tensor([t], dtype=torch.long)
                x_t = sde(x_t, noise, time_steps)
            mean = x_t.mean().item()
            var = x_t.var().item()
            expected_var = None
            if method != "ode":
                marginal_var = hp.get_variance(torch.tensor([num_steps - 1]), method).item()
                if method == "ve":
                    expected_var = marginal_var
                else:
                    expected_var = marginal_var + (1 - marginal_var) * input_var
            print(f"    Mean: {mean:.4f}, Variance: {var:.4f}, Expected Variance: {expected_var if expected_var else 'N/A'}")

            assert x_t.shape == x0.shape, f"Shape mismatch: got {x_t.shape}, expected {x0.shape}"
            assert torch.all(x_t.isfinite()), f"Output contains NaN/Inf for {method}"
            if method == "ve":
                assert abs(mean) < 0.5, f"VE mean too large: {mean}"
                assert abs(
                    var - expected_var) / expected_var < 0.1, f"VE variance mismatch: got {var}, expected {expected_var}"
            elif method == "vp":
                assert abs(mean) < 0.5, f"VP mean too large: {mean}"
                assert abs(
                    var - expected_var) / expected_var < 0.1, f"VP variance mismatch: got {var}, expected {expected_var}"
            elif method == "sub-vp":
                assert abs(mean) < 0.5, f"Sub-VP mean too large: {mean}"
                assert abs(
                    var - expected_var) / expected_var < 0.1, f"Sub-VP variance mismatch: got {var}, expected {expected_var}"
            elif method == "ode":
                if method == "ve":
                    assert torch.allclose(x_t, x0, atol=1e-4), "VE ODE should not change x"
                else:
                    assert x_t.abs().mean() < x0.abs().mean(), "VP/Sub-VP ODE should reduce magnitude"

        print("ForwardSDE tests passed!")

    def test_reverse_sde(self):
        num_steps = 1000
        batch_size = 4
        channels = 3
        height = 32
        width = 32
        hp = HyperParamsSDE(num_steps=num_steps, sigma_end=10.0)

        def mock_score(x, t):
            return -x
        x_T = torch.randn(batch_size, channels, height, width)
        x_T = torch.clamp(x_T, -1, 1)
        for method in ["ve", "vp", "sub-vp", "ode"]:
            print(f"  SDE method: {method}")
            sde = ReverseSDE(hp, method)
            x_t = x_T.clone()
            for t in range(num_steps - 1, -1, -1):
                noise = torch.randn_like(x_t) if method != "ode" else None
                time_steps = torch.tensor([t], dtype=torch.long)
                predicted_noise = mock_score(x_t, time_steps)
                x_t = sde(x_t, noise, predicted_noise, time_steps)
            assert x_t.shape == x_T.shape, f"Shape mismatch: got {x_t.shape}, expected {x_T.shape}"
            assert torch.all(x_t.isfinite()), f"Output contains NaN/Inf for {method}"
            assert x_t.abs().max() < 1e5, f"Output values too large for {method}: max {x_t.abs().max().item()}"
            if method == "ode":
                x_t_1 = x_T.clone()
                x_t_2 = x_T.clone()
                for t in range(num_steps - 1, -1, -1):
                    time_steps = torch.tensor([t], dtype=torch.long)
                    predicted_noise_1 = mock_score(x_t_1, time_steps)
                    predicted_noise_2 = mock_score(x_t_2, time_steps)
                    x_t_1 = sde(x_t_1, None, predicted_noise_1, time_steps)
                    x_t_2 = sde(x_t_2, None, predicted_noise_2, time_steps)
                assert torch.allclose(x_t_1, x_t_2, atol=1e-4), f"ODE not deterministic for {method}"
            print(f"    Final mean: {x_t.mean().item():.4f}, Final variance: {x_t.var().item():.4f}")
        print("ReverseSDE tests passed!")

    def test_sde_training(self):
        torch.manual_seed(42)
        np.random.seed(42)
        noise_p = NoisePredictor(
            in_channels=3,
            down_channels=[16, 32],
            mid_channels=[32, 32],
            up_channels=[32, 16],
            down_sampling=[True, True],
            time_embed_dim=64,
            y_embed_dim=64,
            num_down_blocks=2,
            num_mid_blocks=2,
            num_up_blocks=2,
            down_sampling_factor=2
        )

        cond = TextEncoder(
            use_pretrained_model=True,
            model_name="bert-base-uncased",
            vocabulary_size=30522,
            num_layers=2,
            input_dimension=64,
            output_dimension=64,
            num_heads=2,
            context_length=77
        )
        hp_sde = HyperParamsSDE(num_steps=500, beta_start=1e-4, beta_end=0.02, beta_method="inverse_time")
        opt = torch.optim.Adam(
            [p for p in noise_p.parameters() if p.requires_grad] +
            [p for p in cond.parameters() if p.requires_grad], lr=1e-3
        )
        obj = nn.MSELoss()
        met = Metrics(device="cuda", fid=True, metrics=True, lpips_=True)
        train_dataset = MockDataset(num_samples=10)
        val_dataset = MockDataset(num_samples=5)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2)
        conditional_sde = TrainSDE(
            method="vp",
            noise_predictor=noise_p,
            hyper_params=hp_sde,
            data_loader=train_loader,
            optimizer=opt,
            objective=obj,
            val_loader=val_loader,
            max_epoch=2,
            device="cpu",
            conditional_model=cond,
            metrics_=met,
            store_path="test_sde.pth",
            patience=2,
            warmup_epochs=1,
            val_frequency=1
        )
        unconditional_sde = TrainSDE(
            method="ode",
            noise_predictor=noise_p,
            hyper_params=hp_sde,
            data_loader=train_loader,
            optimizer=opt,
            objective=obj,
            #val_loader=val_loader,
            max_epoch=2,
            device="cpu",
            #conditional_model=cond,
            #metrics_=met,
            store_path="test_sde.pth",
            patience=2,
            warmup_epochs=1,
            val_frequency=1
        )

        try:
            train_losses1, best_val_loss1 = conditional_sde()
            print(f"Training completed. Final train losses: {train_losses1}")
            print(f"Best validation loss: {best_val_loss1:.4f}")

            assert os.path.exists("test_sde.pth"), "Checkpoint file was not saved"

            epoch1, loss1 = conditional_sde.load_checkpoint("test_sde.pth")
            print(f"Loaded checkpoint from epoch {epoch1} with loss {loss1:.4f}")
            assert epoch1 >= 0, "Invalid epoch loaded from checkpoint"
            assert loss1 < float('inf'), "Invalid loss loaded from checkpoint"

            assert len(train_losses1) > 0, "No training losses recorded"
            assert all(isinstance(l, float) for l in train_losses1), "Invalid training losses"
            assert isinstance(best_val_loss1, float), "Invalid best validation loss"

        except Exception as e:
            print(f"Test failed with error: {e}")
            raise

        finally:
            if os.path.exists("test_sde.pth"):
                os.remove("test_sde.pth")
            if os.path.exists("test_sde.pth_early_stop.pth"):
                os.remove("test_sde.pth_early_stop.pth")

        try:
            train_losses2, best_val_loss2 = unconditional_sde()
            print(f"Training completed. Final train losses: {train_losses2}")
            print(f"Best validation loss: {best_val_loss2:.4f}")

            assert os.path.exists("test_sde.pth"), "Checkpoint file was not saved"

            epoch2, loss2 = unconditional_sde.load_checkpoint("test_sde.pth")
            print(f"Loaded checkpoint from epoch {epoch2} with loss {loss2:.4f}")
            assert epoch2 >= 0, "Invalid epoch loaded from checkpoint"
            assert loss2 < float('inf'), "Invalid loss loaded from checkpoint"

            assert len(train_losses2) > 0, "No training losses recorded"
            assert all(isinstance(l, float) for l in train_losses2), "Invalid training losses"
            assert isinstance(best_val_loss2, float), "Invalid best validation loss"

        except Exception as e:
            print(f"Test failed with error: {e}")
            raise

        finally:
            if os.path.exists("test_sde.pth"):
                os.remove("test_sde.pth")
            if os.path.exists("test_sde.pth_early_stop.pth"):
                os.remove("test_sde.pth_early_stop.pth")


    def test_sample_sde(self):

        noise_p = NoisePredictor(
            in_channels=3,
            down_channels=[16, 32],
            mid_channels=[32, 32],
            up_channels=[32, 16],
            down_sampling=[True, True],
            time_embed_dim=32,
            y_embed_dim=32,
            num_down_blocks=1,
            num_mid_blocks=1,
            num_up_blocks=1,
            down_sampling_factor=1
        )

        cond = TextEncoder(
            use_pretrained_model=True,
            model_name="bert-base-uncased",
            vocabulary_size=30522,
            num_layers=2,
            input_dimension=32,
            output_dimension=32,
            num_heads=2,
            context_length=77
        )
        hp_sde = HyperParamsSDE(num_steps=500, beta_start=1e-4, beta_end=0.02, beta_method="linear")
        r_sde = ReverseSDE(hp_sde, "vp")
        sampler_sde = SampleSDE(
            reverse_diffusion=r_sde,
            noise_predictor=noise_p,
            image_shape=(32, 32),
            conditional_model=cond,
            tokenizer="bert-base-uncased",
            max_length=77,
            batch_size=2,
            in_channels=3,
            device="cpu",
            output_range=(-1, 1))
        sampler_sde1 = SampleSDE(
            reverse_diffusion=r_sde,
            noise_predictor=noise_p,
            image_shape=(32, 32),
            conditional_model=None, #cond,
            tokenizer="bert-base-uncased",
            max_length=77,
            batch_size=2,
            in_channels=3,
            device="cpu",
            output_range=(-1, 1))

        img_cond = sampler_sde(
            conditions=["nothing", "something"],
            save_images=True,
            save_path="../sde_generated"
        )
        assert img_cond.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {img_cond.shape}"
        assert torch.all(img_cond >= 0) and torch.all(img_cond <= 1), "Images out of [0, 1] range"
        print("conditional SampleSDE test passed!")


        imgs = sampler_sde1()
        assert imgs.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {imgs.shape}"
        assert torch.all(imgs >= 0) and torch.all(imgs <= 1), "Images out of [0, 1] range"
        print("unconditional SampleSDE test passed!")



if __name__ == '__main__':
    unittest.main()