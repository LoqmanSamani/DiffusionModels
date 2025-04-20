import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import unittest
from ldm import AutoencoderLDM, TrainLDM, TrainAE, SampleLDM
from sde import ForwardSDE, ReverseSDE, HyperParamsSDE
from utils import TextEncoder, NoisePredictor, Metrics


class TestLDM(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=self.transform
        )
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=self.transform
        )
        train_subset_indices = torch.randperm(len(train_dataset))[:40]
        test_subset_indices = torch.randperm(len(test_dataset))[:10]
        train_subset = Subset(train_dataset, train_subset_indices)
        test_subset = Subset(test_dataset, test_subset_indices)
        self.train_loader = DataLoader(train_subset, batch_size=20, shuffle=True)
        self.test_loader = DataLoader(test_subset, batch_size=10, shuffle=False)

    def test_autoencoder(self):
        comp = AutoencoderLDM(
            in_channels=1,
            down_channels=[8, 16, 32],
            up_channels=[32, 16, 8],
            out_channels=1,
            dropout_rate=0.2,
            latent_channels=1,
            num_heads=2,
            num_groups=8,
            num_layers_per_block=2,
            total_down_sampling_factor=2,
            num_embeddings=16
        ).to(self.device)

        met = Metrics(device=self.device, fid=True, metrics=True, lpips_=True)
        opt = torch.optim.Adam(comp.parameters(), lr=1e-3)

        t_auto = TrainAE(
            model=comp,
            optimizer=opt,
            data_loader=self.train_loader,
            val_loader=self.test_loader,
            max_epoch=5,
            metrics_=met,
            device=self.device,
            save_path="../vlc_model.pth",
            checkpoint=3,
            kl_warmup_epochs=2,
            patience=5,
            val_frequency=2
        )
        results = t_auto()

    def test_train_sampling(self):
        comp = AutoencoderLDM(
            in_channels=1,
            down_channels=[8, 16],
            up_channels=[16, 8],
            out_channels=1,
            dropout_rate=0.2,
            latent_channels=1,
            num_heads=1,
            num_groups=8,
            num_layers_per_block=1,
            total_down_sampling_factor=2,
            num_embeddings=16
        ).to(self.device)
        noise_p = NoisePredictor(
            in_channels=1,
            down_channels=[16, 32],
            mid_channels=[32, 32],
            up_channels=[32, 16],
            down_sampling=[True, True],
            time_embed_dim=32,
            y_embed_dim=32,
            num_down_blocks=1,
            num_mid_blocks=1,
            num_up_blocks=1,
            down_sampling_factor=2
        ).to(self.device)

        cond = TextEncoder(
            use_pretrained_model=True,
            model_name="bert-base-uncased",
            vocabulary_size=30522,
            num_layers=2,
            input_dimension=32,
            output_dimension=32,
            num_heads=2,
            context_length=77
        ).to(self.device)
        hp_sde = HyperParamsSDE(
            num_steps=500,
            beta_start=1e-4,
            beta_end=0.02,
            sigma_start=1e-3,
            sigma_end=10.0,
            start=0.0,
            end=1.0,
            beta_method="linear"
        )
        f_sde = ForwardSDE(hp_sde, "ode")
        r_sde = ReverseSDE(hp_sde, "ode")
        opt = torch.optim.Adam(
            [p for p in noise_p.parameters() if p.requires_grad] +
            [p for p in cond.parameters() if p.requires_grad],
            lr=1e-3
        )
        met = Metrics(device=self.device, fid=True, metrics=True, lpips_=True)
        obj = nn.MSELoss()
        t_ldm = TrainLDM(
            model="sde",
            forward_model=f_sde,
            noise_predictor=noise_p,
            hyper_params=hp_sde,
            compressor_model=comp,
            conditional_model=cond,
            reverse_diffusion=r_sde,
            metrics_=met,
            optimizer=opt,
            objective=obj,
            data_loader=self.train_loader,
            val_loader=self.test_loader,
            max_epoch=2,
            device=self.device,
            store_path="../test_ldm.pth",
            val_frequency=2
        )
        sampler_ldm = SampleLDM(
            model="sde",
            reverse_diffusion=r_sde,
            noise_predictor=noise_p,
            compressor_model=comp,
            image_shape=(64, 64),
            conditional_model=cond,
            tokenizer="bert-base-uncased",
            max_length=77,
            batch_size=3,
            in_channels=1,
            device=self.device,
            output_range=(-1, 1)
        )
        train_results = t_ldm()
        conditions = ["nothing", "something", "a cat is playing with a dog"]
        imgs = sampler_ldm(conditions=conditions, save_images=False)



if __name__ == '__main__':
    unittest.main()