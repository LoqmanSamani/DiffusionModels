import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from text_encoder import TextEncoder
from noise_predictor import NoisePredictor





class DummyDataset(Dataset):
    def __init__(self, num_samples=32, image_size=64, vocab_size=30522):
        self.num_samples = num_samples
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        text = torch.randint(0, self.vocab_size, (20,))
        return image, text

def test_unet_and_encoder():

    torch.manual_seed(42)

    batch_size = 32
    image_size = 64
    latent_channels = 3
    vocab_size = 30522
    context_length = 77
    embed_dim = 512

    dataset = DummyDataset(num_samples=batch_size, image_size=image_size, vocab_size=vocab_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    unet = NoisePredictor(
        in_channels=latent_channels,
        down_channels=[64, 128, 256, 512],
        mid_channels=[512, 512, 512],
        up_channels=[512, 256, 128, 64],
        down_sampling=[True, True, True],
        time_embed_dim=128,
        y_embed_dim=embed_dim,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout_rate=0.1
    )

    encoder = TextEncoder(
        use_pretrained_model=False,
        vocabulary_size=vocab_size,
        num_layers=2,
        input_dimension=embed_dim,
        output_dimension=embed_dim,
        num_heads=4,
        context_length=context_length,
        dropout_rate=0.1
    )

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    unet = unet.to(device)
    encoder = encoder.to(device)
    unet.train()
    encoder.train()

    optimizer = torch.optim.Adam(list(unet.parameters()) + list(encoder.parameters()), lr=1e-4)

    print("Testing UNet and Encoder for 1 epoch...")

    for images, texts in data_loader:
        images = images.to(device)
        texts = texts.to(device)

        t = torch.randint(1, 1000, (batch_size,), device=device)

        text_embeddings = encoder(texts)
        print(f"Text embeddings shape: {text_embeddings.shape}")

        noise_pred = unet(images, t, y=text_embeddings)
        print(f"Input shape: {images.shape}")
        print(f"Output (noise_pred) shape: {noise_pred.shape}")

        assert noise_pred.shape == images.shape, f"Expected output shape {images.shape}, got {noise_pred.shape}"

        loss = torch.nn.functional.mse_loss(noise_pred, images)
        print(f"Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        break

    print("Test completed successfully! Models work as expected with correct input/output dimensions.")

if __name__ == "__main__":
    #tic = time.time()
    test_unet_and_encoder()
    #toc = time.time()

    #print("run time: ", toc - tic)