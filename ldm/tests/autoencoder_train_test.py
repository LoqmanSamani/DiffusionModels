import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from autoencoder import LDMAutoencoder as LDMAE
from autoencoder_train import AETrain
import time




def test_vlc_train():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    small_subset = Subset(dataset, range(50))
    data_loader = DataLoader(small_subset, batch_size=10, shuffle=True)


    model = LDMAE(
        in_channels=3,
        down_channels=[32, 64],
        up_channels=[64, 32],
        out_channels=3,
        latent_channels=3,  # match z-shape
        dropout_rate=0.1,
        num_heads=1,
        num_groups=8,
        num_layers_per_block=1,
        total_down_sampling_factor=2,
        num_embeddings=128,
        use_vq=False, # if True, it uses VQ method
        beta=1e-4
    )

    optimizer = optim.Adam(model.parameters(), lr=9e-6)

    trainer = AETrain(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        max_epoch=5,
        device="cpu",  # Use "cuda" if available
        save_path="../vlc_test_model.pth",
        checkpoint=5,
        kl_warmup_epochs=2,
        patience=5,
        perceptual_weight=0.5,
        per_loss=True,
        metrics=True,
        fid=True
    )

    losses, best_val_loss = trainer.train()
    print(f"Final Best Loss: {best_val_loss:.4f}")
    print(f"Training Losses: {[f'{l:.4f}' for l in losses]}")

tic = time.time()
if __name__ == "__main__":
    test_vlc_train()
toc = time.time()
print(toc - tic)