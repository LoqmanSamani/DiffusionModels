import torch
import torch.nn as nn
import unittest
from torch.utils.data import DataLoader, Dataset
from ddpm import ForwardDDPM, ReverseDDPM, HyperParamsDDPM, TrainDDPM, SampleDDPM
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

class TestDDPM(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.batch_size = 2
        self.channels = 3
        self.height = 32
        self.width = 32
        self.num_steps = 1000
        self.tau_num_steps = 100
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.eta = 0.0

        self.hyper_params = HyperParamsDDPM(
            num_steps=self.num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            trainable_beta=False,
            beta_method="linear"
        ).to(self.device)

        self.hyper_params_trainable = HyperParamsDDPM(
            num_steps=self.num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            trainable_beta=True,
            beta_method="linear"
        ).to(self.device)

        self.forward_ddpm = ForwardDDPM(self.hyper_params).to(self.device)
        self.reverse_ddpm = ReverseDDPM(self.hyper_params).to(self.device)
        self.forward_ddpm_trainable = ForwardDDPM(self.hyper_params_trainable).to(self.device)
        self.reverse_ddpm_trainable = ReverseDDPM(self.hyper_params_trainable).to(self.device)

        self.x0 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        self.noise = torch.randn_like(self.x0).to(self.device)

        self.time_steps = torch.randint(1, self.num_steps, (self.batch_size,)).to(self.device)
        self.prev_time_steps = self.time_steps - 1

        self.tau_time_steps = torch.randint(1, self.tau_num_steps, (self.batch_size,)).to(self.device)
        self.tau_prev_time_steps = self.tau_time_steps - 1


    def test_hyper_params_beta_methods(self):

        for method in ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]:
            hyper_params = HyperParamsDDPM(
                num_steps=self.num_steps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                trainable_beta=False,
                beta_method=method
            ).to(self.device)

            self.assertTrue(torch.all(hyper_params.betas >= self.beta_start))
            self.assertTrue(torch.all(hyper_params.betas <= self.beta_end), f"Failed for method {method}")

    def test_hyper_params_invalid_inputs(self):
        with self.assertRaises(ValueError):
            HyperParamsDDPM(beta_start=-0.1, beta_end=0.02)
        with self.assertRaises(ValueError):
            HyperParamsDDPM(beta_start=0.03, beta_end=0.02)
        with self.assertRaises(ValueError):
            HyperParamsDDPM(num_steps=0)
        with self.assertRaises(ValueError):
            HyperParamsDDPM(beta_method="invalid")

    def test_forward_ddpm(self):
        xt = self.forward_ddpm(self.x0, self.noise, self.time_steps)
        self.assertEqual(xt.shape, self.x0.shape)
        sqrt_alpha_bars_t = self.hyper_params.sqrt_alpha_bars[self.time_steps].view(-1, 1, 1, 1).to(self.device)
        sqrt_one_minus_alpha_bars_t = self.hyper_params.sqrt_one_minus_alpha_bars[self.time_steps].view(-1, 1, 1, 1).to(self.device)
        expected_xt = sqrt_alpha_bars_t * self.x0 + sqrt_one_minus_alpha_bars_t * self.noise
        self.assertTrue(torch.allclose(xt, expected_xt, atol=1e-5))
        xt_trainable = self.forward_ddpm_trainable(self.x0, self.noise, self.time_steps)
        _, _, _, sqrt_alpha_bars_t_trainable, sqrt_one_minus_alpha_bars_t_trainable = (
            self.hyper_params_trainable.compute_schedule(self.hyper_params_trainable.betas)
        )
        sqrt_alpha_bars_t_trainable = sqrt_alpha_bars_t_trainable[self.time_steps].view(-1, 1, 1, 1).to(self.device)
        sqrt_one_minus_alpha_bars_t_trainable = sqrt_one_minus_alpha_bars_t_trainable[self.time_steps].view(-1, 1, 1,1).to(self.device)
        expected_xt_trainable = sqrt_alpha_bars_t_trainable * self.x0 + sqrt_one_minus_alpha_bars_t_trainable * self.noise
        self.assertTrue(torch.allclose(xt_trainable, expected_xt_trainable, atol=1e-5))

    def test_forward_ddpm_invalid_time_steps(self):
        invalid_time_steps = torch.tensor([self.num_steps, 0]).to(self.device)
        with self.assertRaises(ValueError):
            self.forward_ddpm(self.x0, self.noise, invalid_time_steps)

    def test_reverse_ddpm(self):
        hyper_params = HyperParamsDDPM(num_steps=1000, beta_method="quadratic")
        reverse = ReverseDDPM(hyper_params)
        xt = torch.randn(4, 3, 32, 32)
        predicted_noise = torch.randn_like(xt)
        time_steps = torch.tensor([0, 250, 500, 750])
        xt_minus_1 = reverse(xt, predicted_noise, time_steps)
        assert xt_minus_1.shape == xt.shape, f"Expected shape {xt.shape}, got {xt_minus_1.shape}"
        print("ReverseDDPM test passed!")


    def test_reverse_ddpm_invalid_time_steps(self):

        invalid_time_steps = torch.tensor([self.num_steps, 0]).to(self.device)
        predicted_noise = torch.randn_like(self.x0).to(self.device)
        with self.assertRaises(ValueError):
            self.reverse_ddpm(self.x0, predicted_noise, invalid_time_steps)

    def test_trainable_beta_consistency(self):
        optimizer = torch.optim.SGD([self.hyper_params_trainable.betas], lr=0.01)
        original_betas = self.hyper_params_trainable.betas.clone().detach()
        for _ in range(10):
            loss = self.hyper_params_trainable.betas.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.hyper_params_trainable.constrain_betas()
        self.assertTrue(torch.all(self.hyper_params_trainable.betas >= self.hyper_params_trainable.beta_start))
        self.assertTrue(torch.all(self.hyper_params_trainable.betas <= self.hyper_params_trainable.beta_end))
        self.assertFalse(torch.allclose(self.hyper_params_trainable.betas, original_betas, atol=1e-3))


    def test_ddpm_training(self):
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
        hp_ddpm = HyperParamsDDPM(num_steps=500, beta_start=1e-4, beta_end=0.02, beta_method="linear")
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
        conditional_ddpm = TrainDDPM(
            noise_predictor=noise_p,
            hyper_params=hp_ddpm,
            data_loader=train_loader,
            optimizer=opt,
            objective=obj,
            val_loader=val_loader,
            max_epoch=2,
            device=self.device,
            conditional_model=cond,
            metrics_=met,
            store_path="test_ddpm.pth",
            patience=2,
            warmup_epochs=1,
            val_frequency=1
        )
        unconditional_ddim = TrainDDPM(
            noise_predictor=noise_p,
            hyper_params=hp_ddpm,
            data_loader=train_loader,
            optimizer=opt,
            objective=obj,
            #val_loader=val_loader,
            max_epoch=2,
            device=self.device,
            #conditional_model=cond,
            #metrics_=met,
            store_path="test_ddpm.pth",
            patience=2,
            warmup_epochs=1,
            val_frequency=1
        )

        try:
            train_losses1, best_val_loss1 = conditional_ddpm()
            print(f"Training completed. Final train losses: {train_losses1}")
            print(f"Best validation loss: {best_val_loss1:.4f}")

            assert os.path.exists("test_ddpm.pth"), "Checkpoint file was not saved"

            epoch1, loss1 = conditional_ddpm.load_checkpoint("test_ddpm.pth")
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
            if os.path.exists("test_ddpm.pth"):
                os.remove("test_ddpm.pth")
            if os.path.exists("test_ddpm.pth_early_stop.pth"):
                os.remove("test_ddpm.pth_early_stop.pth")

        try:
            train_losses2, best_val_loss2 = unconditional_ddim()
            print(f"Training completed. Final train losses: {train_losses2}")
            print(f"Best validation loss: {best_val_loss2:.4f}")

            assert os.path.exists("test_ddpm.pth"), "Checkpoint file was not saved"

            epoch2, loss2 = unconditional_ddim.load_checkpoint("test_ddpm.pth")
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
            if os.path.exists("test_ddpm.pth"):
                os.remove("test_ddpm.pth")
            if os.path.exists("test_ddpm.pth_early_stop.pth"):
                os.remove("test_ddpm.pth_early_stop.pth")

    def test_sample_ddpm_unconditional(self):

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
        hp_ddpm = HyperParamsDDPM(num_steps=500, beta_start=1e-4, beta_end=0.02,
                                  beta_method="sigmoid")
        r_ddpm = ReverseDDPM(hp_ddpm)
        sampler_ddpm = SampleDDPM(
            reverse_diffusion=r_ddpm,
            noise_predictor=noise_p,
            image_shape=(32, 32),
            conditional_model=cond,
            tokenizer="bert-base-uncased",
            max_length=77,
            batch_size=2,
            in_channels=3,
            device=self.device,
            output_range=(-1, 1))
        sampler_ddpm1 = SampleDDPM(
            reverse_diffusion=r_ddpm,
            noise_predictor=noise_p,
            image_shape=(32, 32),
            conditional_model=None, #cond,
            tokenizer="bert-base-uncased",
            max_length=77,
            batch_size=2,
            in_channels=3,
            device=self.device,
            output_range=(-1, 1))

        img_cond = sampler_ddpm(
            conditions=["nothing", "something"],
            save_images=True,
            save_path="../ddpm_generated"
        )
        assert img_cond.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {img_cond.shape}"
        assert torch.all(img_cond >= 0) and torch.all(img_cond <= 1), "Images out of [0, 1] range"
        print("conditional SampleDDPM test passed!")


        imgs = sampler_ddpm1()
        assert imgs.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {imgs.shape}"
        assert torch.all(imgs >= 0) and torch.all(imgs <= 1), "Images out of [0, 1] range"
        print("unconditional SampleDDPM test passed!")


if __name__ == '__main__':
    unittest.main()