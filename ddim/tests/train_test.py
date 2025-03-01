from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
from train import Train
from network import UNet
from config import Config





def test_train():
    device = "cpu"

    train_data = CIFAR10(root="mnist", train=True, download=True, transform=ToTensor())
    small_subset = Subset(train_data, range(300))  # use a small subset for faster testing
    train_loader = DataLoader(small_subset, batch_size=128, shuffle=True)

    model = UNet(
        in_channels=3,
        down_channels=[32, 64, 128, 256],
        mid_channels=[256, 256, 128],
        up_channels=[256, 128, 64, 16],
        down_sampling=[True, True, False],
        time_embed_dim=128,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout_rate=0.2
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    config = Config(
        model_path="test_model.pth",
        output_shape=(3, 32, 32),
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_func=loss_fn,
        train_epochs=2,
        in_channels=3,
        beta_range=(1e-4, 0.02),
        beta_method="linear",
        num_steps=1000,
        num_tau_steps=100,
        device=device,
        eta=0
    )

    trainer = Train(config)
    trainer.fit()


test_train()
