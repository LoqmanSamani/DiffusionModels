import torch
from noise_predictor import NoisePredictor

def test_noise_predictor():
    # Define model parameters
    in_channels = 3
    down_channels = [32, 64, 128, 256, 512]
    mid_channels = [512, 512, 512, 512]
    up_channels = [512, 256, 128, 64, 32]
    down_sampling = [True, True, True, True]
    time_embed_dim = 128
    y_embed_dim = 768
    num_down_blocks = 2
    num_mid_blocks = 2
    num_up_blocks = 2

    # Initialize model
    model = NoisePredictor(
        in_channels=in_channels,
        down_channels=down_channels,
        mid_channels=mid_channels,
        up_channels=up_channels,
        down_sampling=down_sampling,
        time_embed_dim=time_embed_dim,
        y_embed_dim=y_embed_dim,
        num_down_blocks=num_down_blocks,
        num_mid_blocks=num_mid_blocks,
        num_up_blocks=num_up_blocks,
        dropout_rate=0.1,
        down_sampling_factor=2,
        where_y=True,
        y_to_all=False
    )
    model.initialize_weights()

    # Create sample inputs
    batch_size = 2
    height, width = 512, 512
    x = torch.randn(batch_size, in_channels, height, width)
    t = torch.randint(0, 1000, (batch_size,))
    y = torch.randn(batch_size, y_embed_dim)

    # Move to GPU if available
    device = "cpu"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)
    t = t.to(device)
    y = y.to(device)

    # Forward pass
    try:
        output = model(x, t, y)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == x.shape, "Output shape should match input shape"
        print("Test passed: Model runs successfully and preserves input shape.")
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    test_noise_predictor()