from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
from train import Train
from network import DDPMpp
from config import Config





def test_train():
    device = "cpu"

    train_data = CIFAR10(root="mnist", train=True, download=True, transform=ToTensor())
    small_subset = Subset(train_data, range(60))  # use a small subset for faster testing
    train_loader = DataLoader(small_subset, batch_size=20, shuffle=True)

    methods = ["ve", "vp", "sub-vp"]

    for method in methods:

        config = Config(
            in_channels=3,
            down_channels=[32, 64, 128, 256],
            mid_channels=[256, 256, 128],
            up_channels=[256, 128, 64, 16],
            down_sampling=[True, True, False],
            num_groups=8,
            embed_dim=128,
            num_down_blocks=2,
            num_mid_blocks=2,
            num_up_blocks=2,
            dropout_rate=0.3,
            num_attention_heads=2,
            down_sampling_factor=2,
            upsampling_factor=2,
            apply_down_conv=True,
            apply_down_pool=True,
            apply_up_conv=True,
            kernel_size=3,
            norm=True,
            activation=True,
            method=method,
            start=None,
            end=None,
            max_steps=400,
            sigma_min=None,
            sigma_max=None,
            beta_range=None,
            beta_schedule_method="linear",
            max_epoch=5,
            device=None,
            optimizer=None,
            objective=None,
            save_path=None,
            checkpoint=None
        )

        model = DDPMpp(config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        objective = nn.MSELoss()
        trainer = Train(config, model, train_loader, optimizer, objective, device)
        losses, best_loss = trainer.train()
        print(method)
        print("---"*20)
        print(losses, best_loss)


test_train()