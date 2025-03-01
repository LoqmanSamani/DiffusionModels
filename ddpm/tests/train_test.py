import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from config import Config
from network import UNet
from train import Train






def test_training():
    """tests function to train the model on a small MNIST subset"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)  # normalize to range [-1, 1]
    ])

    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    small_subset = Subset(dataset, range(300))
    train_loader = DataLoader(small_subset, batch_size=128, shuffle=True)

    model = UNet(
        in_channels=1,
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

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    config = Config(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss=loss_fn,
        model_path="./test_model.pth",
        num_epochs=2,
        num_diffusion_steps=1000,
        num_time_steps=1000,
        device=device
    )

    trainer = Train(config)
    trainer.fit()
    print("Test training completed successfully!")


test_training()

