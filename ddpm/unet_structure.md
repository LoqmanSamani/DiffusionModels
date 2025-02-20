# UNet Model Structure

**(Batch Size: 2, Image Size: 64x64, Input Channels: 3)**

## 1. Initial Convolution

    conv1 → Conv2D(3 → 32) → (2, 32, 64, 64)

## 2. Down Blocks

### Down Block 0 (n iterations)
    (GroupNorm(8) → SiLU → Conv2D(32 → 64))
    Add Time Embedding
    (GroupNorm(8) → SiLU → Conv2D(64 → 64))
    ResNet Connection
    Self-Attention
    Downsampling (Conv2D + MaxPool → concat) → (2, 64, 32, 32)

### Down Block 1 (n iterations)
    (GroupNorm(8) → SiLU → Conv2D(64 → 128))
    Add Time Embedding
    (GroupNorm(8) → SiLU → Conv2D(128 → 128))
    ResNet Connection
    Self-Attention
    Downsampling (Conv2D + MaxPool → concat) → (2, 128, 16, 16)

### Down Block 2 (n iterations)
    (GroupNorm(8) → SiLU → Conv2D(128 → 256))
    Add Time Embedding
    (GroupNorm(8) → SiLU → Conv2D(256 → 256))
    ResNet Connection
    Self-Attention
    No Downsampling (False) → (2, 256, 16, 16)

## 3. Middle Blocks

### Mid Block 0 (n iterations)
    (GroupNorm(8) → SiLU → Conv2D(256 → 256))
    Add Time Embedding
    ResNet Connection
    Self-Attention → (2, 256, 16, 16)

### Mid Block 1 (n iterations)
    (GroupNorm(8) → SiLU → Conv2D(256 → 128))
    Add Time Embedding
    ResNet Connection
    Self-Attention → (2, 128, 16, 16)

## 4. Up Blocks

### Up Block 0 (n iterations)
    Upsampling (ConvTranspose2D + Upsample → concat) → (2, 128, 16, 16)
    (GroupNorm(8) → SiLU → Conv2D(256 → 128))
    Add Time Embedding
    ResNet Connection
    Self-Attention

### Up Block 1 (n iterations)
    Upsampling (ConvTranspose2D + Upsample → concat) → (2, 64, 32, 32)
    (GroupNorm(8) → SiLU → Conv2D(128 → 64))
    Add Time Embedding
    ResNet Connection
    Self-Attention

### Up Block 2 (n iterations)
    Upsampling (ConvTranspose2D + Upsample → concat) → (2, 16, 64, 64)
    (GroupNorm(8) → SiLU → Conv2D(64 → 16))
    Add Time Embedding
    ResNet Connection
    Self-Attention

## 5. Final Convolution

    conv2 → (GroupNorm(8) → Conv2D(16 → 3)) → (2, 3, 64, 64)

