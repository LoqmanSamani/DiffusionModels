import torch
import torch.nn as nn
import unittest
from torch.utils.data import DataLoader, Dataset
from ddim import ForwardDDIM, ReverseDDIM, HyperParamsDDIM, TrainDDIM, SampleDDIM
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

class TestDDIM(unittest.TestCase):
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

        self.hyper_params = HyperParamsDDIM(
            eta=self.eta,
            num_steps=self.num_steps,
            tau_num_steps=self.tau_num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            trainable_beta=False,
            beta_method="linear"
        ).to(self.device)

        self.hyper_params_trainable = HyperParamsDDIM(
            eta=self.eta,
            num_steps=self.num_steps,
            tau_num_steps=self.tau_num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            trainable_beta=True,
            beta_method="linear"
        ).to(self.device)

        self.forward_ddim = ForwardDDIM(self.hyper_params).to(self.device)
        self.reverse_ddim = ReverseDDIM(self.hyper_params).to(self.device)
        self.forward_ddim_trainable = ForwardDDIM(self.hyper_params_trainable).to(self.device)
        self.reverse_ddim_trainable = ReverseDDIM(self.hyper_params_trainable).to(self.device)

        self.x0 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        self.noise = torch.randn_like(self.x0).to(self.device)

        self.time_steps = torch.randint(1, self.num_steps, (self.batch_size,)).to(self.device)
        self.prev_time_steps = self.time_steps - 1

        self.tau_time_steps = torch.randint(1, self.tau_num_steps, (self.batch_size,)).to(self.device)
        self.tau_prev_time_steps = self.tau_time_steps - 1

    def test_hyper_params_initialization(self):

        self.assertTrue(torch.all(self.hyper_params.betas >= self.beta_start))
        self.assertTrue(torch.all(self.hyper_params.betas <= self.beta_end))
        self.assertEqual(self.hyper_params.betas.shape, (self.num_steps,))
        alphas = 1 - self.hyper_params.betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.assertTrue(torch.allclose(self.hyper_params.alphas, alphas))
        self.assertTrue(torch.allclose(self.hyper_params.alpha_cumprod, alpha_cumprod))
        self.assertTrue(torch.allclose(self.hyper_params.sqrt_alpha_cumprod, torch.sqrt(alpha_cumprod)))
        self.assertTrue(torch.allclose(self.hyper_params.sqrt_one_minus_alpha_cumprod, torch.sqrt(1 - alpha_cumprod)))
        self.assertEqual(self.hyper_params.tau_indices.shape, (self.tau_num_steps,))
        self.assertTrue(torch.all(self.hyper_params.tau_indices >= 0))
        self.assertTrue(torch.all(self.hyper_params.tau_indices < self.num_steps))
        self.assertTrue(isinstance(self.hyper_params_trainable.betas, nn.Parameter))
        self.assertFalse(hasattr(self.hyper_params_trainable, 'alphas'))

    def test_hyper_params_get_tau_schedule(self):

        tau_betas, tau_alphas, tau_alpha_cumprod, tau_sqrt_alpha_cumprod, tau_sqrt_one_minus_alpha_cumprod = (self.hyper_params.get_tau_schedule())
        expected_betas = self.hyper_params.betas[self.hyper_params.tau_indices]
        expected_alpha_cumprod = self.hyper_params.alpha_cumprod[self.hyper_params.tau_indices]
        self.assertTrue(torch.allclose(tau_betas, expected_betas))
        self.assertTrue(torch.allclose(tau_alpha_cumprod, expected_alpha_cumprod))
        self.assertTrue(torch.allclose(tau_sqrt_alpha_cumprod, torch.sqrt(tau_alpha_cumprod)))
        self.assertTrue(torch.allclose(tau_sqrt_one_minus_alpha_cumprod, torch.sqrt(1 - tau_alpha_cumprod)))
        tau_betas_trainable, _, _, _, _ = self.hyper_params_trainable.get_tau_schedule()
        expected_betas_trainable = self.hyper_params_trainable.betas[self.hyper_params_trainable.tau_indices]
        self.assertTrue(torch.allclose(tau_betas_trainable, expected_betas_trainable))

    def test_hyper_params_beta_methods(self):

        for method in ["linear", "sigmoid", "quadratic", "constant", "inverse_time"]:
            hyper_params = HyperParamsDDIM(
                eta=self.eta,
                num_steps=self.num_steps,
                tau_num_steps=self.tau_num_steps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                trainable_beta=False,
                beta_method=method
            ).to(self.device)

            self.assertTrue(torch.all(hyper_params.betas >= self.beta_start))
            self.assertTrue(torch.all(hyper_params.betas <= self.beta_end), f"Failed for method {method}")

    def test_hyper_params_invalid_inputs(self):
        with self.assertRaises(ValueError):
            HyperParamsDDIM(beta_start=-0.1, beta_end=0.02)
        with self.assertRaises(ValueError):
            HyperParamsDDIM(beta_start=0.03, beta_end=0.02)
        with self.assertRaises(ValueError):
            HyperParamsDDIM(num_steps=0)
        with self.assertRaises(ValueError):
            HyperParamsDDIM(beta_method="invalid")

    def test_forward_ddim(self):
        xt = self.forward_ddim(self.x0, self.noise, self.time_steps)
        self.assertEqual(xt.shape, self.x0.shape)
        sqrt_alpha_cumprod_t = self.hyper_params.sqrt_alpha_cumprod[self.time_steps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.hyper_params.sqrt_one_minus_alpha_cumprod[self.time_steps].view(-1, 1, 1, 1)
        expected_xt = sqrt_alpha_cumprod_t * self.x0 + sqrt_one_minus_alpha_cumprod_t * self.noise
        self.assertTrue(torch.allclose(xt, expected_xt, atol=1e-5))
        xt_trainable = self.forward_ddim_trainable(self.x0, self.noise, self.time_steps)
        _, _, _, sqrt_alpha_cumprod_t_trainable, sqrt_one_minus_alpha_cumprod_t_trainable = (self.hyper_params_trainable.compute_schedule(self.hyper_params_trainable.betas))
        sqrt_alpha_cumprod_t_trainable = sqrt_alpha_cumprod_t_trainable[self.time_steps].view(-1, 1, 1, 1).to(self.device)
        sqrt_one_minus_alpha_cumprod_t_trainable = sqrt_one_minus_alpha_cumprod_t_trainable[self.time_steps].view(-1, 1, 1, 1).to(self.device)
        expected_xt_trainable = sqrt_alpha_cumprod_t_trainable * self.x0 + sqrt_one_minus_alpha_cumprod_t_trainable * self.noise
        self.assertTrue(torch.allclose(xt_trainable, expected_xt_trainable, atol=1e-5))

    def test_forward_ddim_invalid_time_steps(self):
        invalid_time_steps = torch.tensor([self.num_steps, 0]).to(self.device)
        with self.assertRaises(ValueError):
            self.forward_ddim(self.x0, self.noise, invalid_time_steps)

    def test_reverse_ddim_deterministic(self):
        predicted_noise = torch.randn_like(self.x0).to(self.device)
        xt_prev, x0_pred = self.reverse_ddim(self.x0, predicted_noise, self.tau_time_steps, self.tau_prev_time_steps)
        self.assertEqual(xt_prev.shape, self.x0.shape)
        self.assertEqual(x0_pred.shape, self.x0.shape)
        tau_sqrt_alpha_cumprod_t = self.hyper_params.get_tau_schedule()[3][self.tau_time_steps].to(self.device).view(-1, 1, 1, 1)
        tau_sqrt_one_minus_alpha_cumprod_t = self.hyper_params.get_tau_schedule()[4][self.tau_time_steps].to(self.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_alpha_cumprod_t = self.hyper_params.get_tau_schedule()[3][self.tau_prev_time_steps].to(self.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_one_minus_alpha_cumprod_t = self.hyper_params.get_tau_schedule()[4][self.tau_prev_time_steps].to(self.device).view(-1, 1, 1, 1)
        expected_x0 = (self.x0 - tau_sqrt_one_minus_alpha_cumprod_t * predicted_noise) / tau_sqrt_alpha_cumprod_t
        self.assertTrue(torch.allclose(x0_pred, expected_x0, atol=1e-5))
        direction_coeff = prev_tau_sqrt_one_minus_alpha_cumprod_t
        expected_xt_prev = prev_tau_sqrt_alpha_cumprod_t * expected_x0 + direction_coeff * predicted_noise
        self.assertTrue(torch.allclose(xt_prev, expected_xt_prev, atol=1e-5))
        xt_prev_trainable, x0_pred_trainable = self.reverse_ddim_trainable(self.x0, predicted_noise, self.tau_time_steps, self.tau_prev_time_steps)
        tau_schedule_trainable = self.hyper_params_trainable.get_tau_schedule()
        tau_sqrt_alpha_cumprod_t_trainable = tau_schedule_trainable[3][self.tau_time_steps].to(self.device).view(-1, 1, 1, 1)
        tau_sqrt_one_minus_alpha_cumprod_t_trainable = tau_schedule_trainable[4][self.tau_time_steps].to(self.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_alpha_cumprod_t_trainable = tau_schedule_trainable[3][self.tau_prev_time_steps].to(self.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_one_minus_alpha_cumprod_t_trainable = tau_schedule_trainable[4][self.tau_prev_time_steps].to(self.device).view(-1, 1, 1, 1)
        expected_x0_trainable = (self.x0 - tau_sqrt_one_minus_alpha_cumprod_t_trainable * predicted_noise) / tau_sqrt_alpha_cumprod_t_trainable
        expected_xt_prev_trainable = (prev_tau_sqrt_alpha_cumprod_t_trainable * expected_x0_trainable + prev_tau_sqrt_one_minus_alpha_cumprod_t_trainable * predicted_noise)
        self.assertTrue(torch.allclose(x0_pred_trainable, expected_x0_trainable, atol=1e-5))
        self.assertTrue(torch.allclose(xt_prev_trainable, expected_xt_prev_trainable, atol=1e-5))

    def test_reverse_ddim_stochastic(self):

        hyper_params_stochastic = HyperParamsDDIM(
            eta=1.0,
            num_steps=self.num_steps,
            tau_num_steps=self.tau_num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            trainable_beta=False,
            beta_method="linear"
        ).to(self.device)
        reverse_ddim_stochastic = ReverseDDIM(hyper_params_stochastic).to(self.device)
        predicted_noise = torch.randn_like(self.x0).to(self.device)
        xt_prev, x0_pred = reverse_ddim_stochastic(self.x0, predicted_noise, self.tau_time_steps, self.tau_prev_time_steps)
        self.assertEqual(xt_prev.shape, self.x0.shape)
        self.assertEqual(x0_pred.shape, self.x0.shape)
        tau_sqrt_alpha_cumprod_t = hyper_params_stochastic.get_tau_schedule()[3][self.tau_time_steps].to(self.device).view(-1, 1, 1, 1)
        tau_sqrt_one_minus_alpha_cumprod_t = hyper_params_stochastic.get_tau_schedule()[4][self.tau_time_steps].to(self.device).view(-1, 1, 1, 1)
        expected_x0 = (self.x0 - tau_sqrt_one_minus_alpha_cumprod_t * predicted_noise) / tau_sqrt_alpha_cumprod_t
        self.assertTrue(torch.allclose(x0_pred, expected_x0, atol=1e-5))
        prev_tau_sqrt_alpha_cumprod_t = hyper_params_stochastic.get_tau_schedule()[3][self.tau_prev_time_steps].to(self.device).view(-1, 1, 1, 1)
        prev_tau_sqrt_one_minus_alpha_cumprod_t = hyper_params_stochastic.get_tau_schedule()[4][self.tau_prev_time_steps].to(self.device).view(-1, 1, 1, 1)
        noise_coeff = (tau_sqrt_one_minus_alpha_cumprod_t / prev_tau_sqrt_alpha_cumprod_t) * prev_tau_sqrt_one_minus_alpha_cumprod_t / tau_sqrt_one_minus_alpha_cumprod_t
        direction_coeff = torch.sqrt(prev_tau_sqrt_one_minus_alpha_cumprod_t ** 2 - noise_coeff ** 2)
        expected_xt_prev_mean = prev_tau_sqrt_alpha_cumprod_t * expected_x0 + direction_coeff * predicted_noise

        xt_prev_samples = []
        for _ in range(100):
            xt_prev_sample, _ = reverse_ddim_stochastic(self.x0, predicted_noise, self.tau_time_steps, self.tau_prev_time_steps)
            xt_prev_samples.append(xt_prev_sample)
        xt_prev_mean = torch.mean(torch.stack(xt_prev_samples), dim=0)
        #self.assertTrue(torch.allclose(xt_prev_mean, expected_xt_prev_mean, atol=1e-2))

    def test_reverse_ddim_invalid_time_steps(self):

        invalid_time_steps = torch.tensor([self.tau_num_steps, 0]).to(self.device)
        predicted_noise = torch.randn_like(self.x0).to(self.device)
        with self.assertRaises(ValueError):
            self.reverse_ddim(self.x0, predicted_noise, invalid_time_steps, self.tau_prev_time_steps)

    def test_trainable_beta_consistency(self):

        optimizer = torch.optim.SGD([self.hyper_params_trainable.betas], lr=0.01)
        original_betas = self.hyper_params_trainable.betas.clone().detach()
        for _ in range(10):
            loss = self.hyper_params_trainable.betas.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.hyper_params_trainable.constrain_betas()

        tau_betas, _, _, _, _ = self.hyper_params_trainable.get_tau_schedule()
        expected_tau_betas = self.hyper_params_trainable.betas[self.hyper_params_trainable.tau_indices]
        self.assertTrue(torch.allclose(tau_betas, expected_tau_betas, atol=1e-5))
        self.assertFalse(torch.allclose(tau_betas, original_betas[self.hyper_params_trainable.tau_indices], atol=1e-3))

    def test_deterministic_consistency_across_steps(self):

        hyper_params_50 = HyperParamsDDIM(
            eta=0.0,
            num_steps=self.num_steps,
            tau_num_steps=50,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            trainable_beta=False,
            beta_method="linear"
        ).to(self.device)
        hyper_params_100 = HyperParamsDDIM(
            eta=0.0,
            num_steps=self.num_steps,
            tau_num_steps=100,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            trainable_beta=False,
            beta_method="linear"
        ).to(self.device)

        reverse_ddim_50 = ReverseDDIM(hyper_params_50).to(self.device)
        reverse_ddim_100 = ReverseDDIM(hyper_params_100).to(self.device)

        xt = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        predicted_noise = torch.randn_like(xt).to(self.device)

        tau_step_50 = 49
        tau_step_100 = 99
        prev_tau_step_50 = tau_step_50 - 1
        prev_tau_step_100 = tau_step_100 - 1

        xt_prev_50, x0_50 = reverse_ddim_50(
            xt, predicted_noise,
            torch.full((self.batch_size,), tau_step_50, device=self.device),
            torch.full((self.batch_size,), prev_tau_step_50, device=self.device)
        )
        xt_prev_100, x0_100 = reverse_ddim_100(
            xt, predicted_noise,
            torch.full((self.batch_size,), tau_step_100, device=self.device),
            torch.full((self.batch_size,), prev_tau_step_100, device=self.device)
        )

        self.assertTrue(torch.allclose(x0_50, x0_100, atol=1e-2))

    def test_ddim_training(self):
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
        hp_ddim = HyperParamsDDIM(num_steps=500, tau_num_steps=100, beta_start=1e-4, beta_end=0.02, beta_method="linear")
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
        conditional_ddim = TrainDDIM(
            noise_predictor=noise_p,
            hyper_params=hp_ddim,
            data_loader=train_loader,
            optimizer=opt,
            objective=obj,
            val_loader=val_loader,
            max_epoch=2,
            device=self.device,
            conditional_model=cond,
            metrics_=met,
            store_path="test_ddim.pth",
            patience=2,
            warmup_epochs=1,
            val_frequency=1
        )
        unconditional_ddim = TrainDDIM(
            noise_predictor=noise_p,
            hyper_params=hp_ddim,
            data_loader=train_loader,
            optimizer=opt,
            objective=obj,
            #val_loader=val_loader,
            max_epoch=2,
            device=self.device,
            #conditional_model=cond,
            #metrics_=met,
            store_path="test_ddim.pth",
            patience=2,
            warmup_epochs=1,
            val_frequency=1
        )

        try:
            train_losses1, best_val_loss1 = conditional_ddim()
            print(f"Training completed. Final train losses: {train_losses1}")
            print(f"Best validation loss: {best_val_loss1:.4f}")

            assert os.path.exists("test_ddim.pth"), "Checkpoint file was not saved"

            epoch1, loss1 = conditional_ddim.load_checkpoint("test_ddim.pth")
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
            if os.path.exists("test_ddim.pth"):
                os.remove("test_ddim.pth")
            if os.path.exists("test_ddim.pth_early_stop.pth"):
                os.remove("test_ddim.pth_early_stop.pth")

        try:
            train_losses2, best_val_loss2 = unconditional_ddim()
            print(f"Training completed. Final train losses: {train_losses2}")
            print(f"Best validation loss: {best_val_loss2:.4f}")

            assert os.path.exists("test_ddim.pth"), "Checkpoint file was not saved"

            epoch2, loss2 = unconditional_ddim.load_checkpoint("test_ddim.pth")
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
            if os.path.exists("test_ddim.pth"):
                os.remove("test_ddim.pth")
            if os.path.exists("test_ddim.pth_early_stop.pth"):
                os.remove("test_ddim.pth_early_stop.pth")

    def test_sample_ddim_unconditional(self):

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
        hp_ddim = HyperParamsDDIM(num_steps=500, tau_num_steps=100, beta_start=1e-4, beta_end=0.02,
                                  beta_method="linear")
        r_ddim = ReverseDDIM(hp_ddim)
        sampler_ddim = SampleDDIM(
            reverse_diffusion=r_ddim,
            noise_predictor=noise_p,
            image_shape=(32, 32),
            conditional_model=cond,
            tokenizer="bert-base-uncased",
            max_length=77,
            batch_size=2,
            in_channels=3,
            device=self.device,
            output_range=(-1, 1))
        sampler_ddim1 = SampleDDIM(
            reverse_diffusion=r_ddim,
            noise_predictor=noise_p,
            image_shape=(32, 32),
            conditional_model=None, #cond,
            tokenizer="bert-base-uncased",
            max_length=77,
            batch_size=2,
            in_channels=3,
            device=self.device,
            output_range=(-1, 1))

        img_cond = sampler_ddim(
            conditions=["nothing", "something"],
            save_images=True,
            save_path="ddim_generated"
        )
        assert img_cond.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {img_cond.shape}"
        assert torch.all(img_cond >= 0) and torch.all(img_cond <= 1), "Images out of [0, 1] range"
        print("conditional SampleDDIM test passed!")


        imgs = sampler_ddim1()
        assert imgs.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {imgs.shape}"
        assert torch.all(imgs >= 0) and torch.all(imgs <= 1), "Images out of [0, 1] range"
        print("unconditional SampleDDIM test passed!")


if __name__ == '__main__':
    unittest.main()