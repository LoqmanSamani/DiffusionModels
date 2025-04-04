import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from autoencoder import LDMAutoencoder



def test_ldm_autoencoder():
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    small_subset = Subset(dataset, range(5))
    data_loader = DataLoader(small_subset, batch_size=2, shuffle=False)

    model = LDMAutoencoder(
        in_channels=3,
        down_channels=[32, 64],
        up_channels=[64, 32],
        out_channels=3,
        latent_channels=3,
        dropout_rate=0.1,
        num_heads=2,
        num_groups=8,
        num_layers_per_block=2,
        total_down_sampling_factor=2,
        use_vq=False,
        num_embeddings=128,
        beta=1e-4
    )

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=9e-6)

    print("Testing VariationalLatentCompressor for 1 epoch...")

    for x, _ in data_loader:
        x = x.to(device)
        print(f"Input shape: {x.shape}")

        x_hat, total_loss, reg_loss, z = model(x)

        print(f"Output (x_hat) shape: {x_hat.shape}")
        expected_z_shape = (x.shape[0], 3, 32 // 2, 32 // 2)
        print(f"Latent (z) shape: {z.shape}")
        assert x_hat.shape == x.shape, f"Expected x_hat shape {x.shape}, got {x_hat.shape}"
        assert z.shape == expected_z_shape, f"Expected z shape {expected_z_shape}, got {z.shape}"

        print(f"Total Loss: {total_loss.item():.4f}")
        print(f"KL Loss: {reg_loss.item():.6f}")
        assert total_loss.item() >= 0, "Total loss should be non-negative"
        assert reg_loss.item() >= 0, "KL loss should be non-negative"

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        recon_loss = F.mse_loss(x_hat, x)
        print(f"Reconstruction Loss (MSE): {recon_loss.item():.4f}")
        assert abs(total_loss.item() - (recon_loss.item() + reg_loss.item())) < 1e-5, \
            "Total loss should be the sum of reconstruction and KL loss"

        break

    print("Test completed successfully! Model works as expected with correct input/output dimensions.")


if __name__ == "__main__":
    test_ldm_autoencoder()