import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from vlc import VariationalLatentCompressor as VLC
from train import VLCTrain





def test_vlc_train():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    small_subset = Subset(dataset, range(100))  # Using only 100 samples
    data_loader = DataLoader(small_subset, batch_size=10, shuffle=True)

    model = VLC(
        in_channels=3,
        down_channels=[32, 64],
        up_channels=[64, 32],
        out_channels=3,
        dropout_rate=0.1,
        num_heads=4,
        num_groups=8,
        levels=2,
        down_sampling_factor=2,
        beta=0.1
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = VLCTrain(
        model=model,
        optimizer=optimizer,
        objective=criterion,
        data_loader=data_loader,
        max_epoch=10,  # Train for a few epochs
        device="cpu",
        save_path="vlc_test_model.pth",
        checkpoint=2
    )
    losses, best_loss = trainer.train()
    print(f"Final Loss: {best_loss:.4f}")


test_vlc_train()